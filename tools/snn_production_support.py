#!/usr/bin/env python3
"""Production-run durability helpers for the SNN meta-optimizer.

This module deliberately has no dependency on the experiment implementation.
It provides the small pieces of process and storage infrastructure needed by a
long GPU run: durable event logging, an exclusive run lock, cooperative signal
handling, and checksummed immutable checkpoint generations.

Checkpoint files are local, trusted PyTorch files.  A SHA256 sidecar detects
accidental truncation/corruption; it is not intended to authenticate files
against a malicious writer with access to the run directory.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _datetime
import enum
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import tempfile
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor


EVENT_SCHEMA = "snn.production.event"
EVENT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "snn.production.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
LATEST_SCHEMA = "snn.production.latest"
LATEST_SCHEMA_VERSION = 1

T = TypeVar("T")


def utc_timestamp() -> str:
    """Return a sortable, timezone-explicit UTC timestamp."""
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _jsonable(value: Any) -> Any:
    """Convert common configuration/metric types to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float cannot be logged: {value!r}")
        return value
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError("only scalar tensors may be encoded as JSON")
        return _jsonable(value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    # NumPy scalar types and similar metric wrappers expose item().  Do not use
    # this escape hatch for arbitrary arrays: those would make event lines huge.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        item = item_method()
        if item is not value:
            return _jsonable(item)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def stable_config_digest(config: Any) -> str:
    """Hash a configuration using canonical, strict JSON serialization."""
    encoded = json.dumps(
        _jsonable(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Durably commit directory-entry changes where the platform supports it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(directory), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, *, mode: int = 0o644
) -> None:
    """Write strict JSON using temp+fsync+replace and fsync its directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        output = os.fdopen(fd, "wb", closefd=True)
        fd = -1
        with output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class EventLogger:
    """Append-only JSONL event stream with a stable envelope.

    ``os.write`` avoids userspace buffering.  Critical events additionally call
    ``fsync`` before returning.  ``fsync_all=True`` is useful for debugging at
    the cost of substantially more storage latency.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        run_id: str,
        *,
        context: Mapping[str, Any] | None = None,
        fsync_all: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.context = _jsonable(dict(context or {}))
        self.fsync_all = bool(fsync_all)
        self._lock = threading.RLock()
        self._closed = False
        self._sequence = self._recover_sequence()
        self._fd = os.open(
            str(self.path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
        )

    def _recover_sequence(self) -> int:
        if not self.path.exists():
            return 0
        # Read a bounded tail.  Event sequence numbers are diagnostic ordering,
        # not correctness state; this avoids an O(file-size) resume operation.
        try:
            with self.path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                size = source.tell()
                source.seek(max(0, size - 4 * 1024 * 1024))
                tail = source.read().splitlines()
            for line in reversed(tail):
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if event.get("run_id") == self.run_id:
                    return int(event.get("sequence", 0))
        except OSError:
            pass
        return 0

    def log(
        self, event: str, *, level: str = "info", critical: bool = False, **data: Any
    ) -> dict[str, Any]:
        """Append one event and return the exact envelope that was written."""
        if not event:
            raise ValueError("event must be non-empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("event logger is closed")
            sequence = self._sequence + 1
            envelope = {
                "schema": EVENT_SCHEMA,
                "schema_version": EVENT_SCHEMA_VERSION,
                "timestamp_utc": utc_timestamp(),
                "run_id": self.run_id,
                "sequence": sequence,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "level": str(level),
                "event": str(event),
                "critical": bool(critical),
                "context": self.context,
                "data": _jsonable(data),
            }
            line = (
                json.dumps(
                    envelope,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            view = memoryview(line)
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    raise OSError("short write to event log")
                view = view[written:]
            if critical or self.fsync_all:
                os.fsync(self._fd)
            self._sequence = sequence
            return envelope

    def exception(
        self, event: str, error: BaseException, *, critical: bool = True, **data: Any
    ) -> dict[str, Any]:
        """Log a structured exception, including its formatted traceback."""
        return self.log(
            event,
            level="error",
            critical=critical,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback="".join(traceback.format_exception(error)),
            **data,
        )

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                os.fsync(self._fd)
            finally:
                os.close(self._fd)
                self._closed = True

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class RunLockError(RuntimeError):
    """Raised when another process owns a run lock."""


class RunLock:
    """Advisory, process-lifetime exclusive lock implemented with ``flock``."""

    def __init__(
        self, path: str | os.PathLike[str], *, run_id: str | None = None
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._fd: int | None = None
        self._acquired_at_utc: str | None = None

    def _write_owner_record(self) -> None:
        if self._fd is None:
            raise RuntimeError("run lock is not acquired")
        owner_record = {
            "schema": "snn.production.run_lock",
            "schema_version": 1,
            "acquired_at_utc": self._acquired_at_utc or utc_timestamp(),
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
        }
        encoded = (
            json.dumps(owner_record, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, encoded)
        os.fsync(self._fd)

    def acquire(self) -> "RunLock":
        if self._fd is not None:
            raise RuntimeError("run lock is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                owner = os.read(fd, 64 * 1024).decode("utf-8", errors="replace").strip()
            finally:
                os.close(fd)
            detail = f"; owner={owner}" if owner else ""
            raise RunLockError(
                f"run directory is already locked by another process{detail}"
            ) from error

        self._fd = fd
        self._acquired_at_utc = utc_timestamp()
        try:
            self._write_owner_record()
        except BaseException:
            self.release()
            raise
        return self

    def set_run_id(self, run_id: str) -> None:
        """Durably attach the run identity after bootstrap under the lock."""
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_id = str(run_id)
        self._write_owner_record()

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        self._acquired_at_utc = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class ForcedTermination(SystemExit):
    """Raised on a second termination signal when immediate exit is enabled."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(128 + signum)


class SignalController:
    """Translate Unix signals into checkpoint/stop requests.

    SIGUSR1 requests an immediate checkpoint without stopping.  The first
    SIGINT or SIGTERM requests a checkpoint and graceful shutdown.  A second
    termination signal sets ``force_requested`` and, by default, raises
    :class:`ForcedTermination` immediately.
    """

    def __init__(
        self,
        *,
        on_graceful: Callable[[int], None] | None = None,
        on_forced: Callable[[int], None] | None = None,
        on_checkpoint: Callable[[int], None] | None = None,
        raise_on_second: bool = True,
    ) -> None:
        self.on_graceful = on_graceful
        self.on_forced = on_forced
        self.on_checkpoint = on_checkpoint
        self.raise_on_second = bool(raise_on_second)
        self.stop_requested = threading.Event()
        self.checkpoint_requested = threading.Event()
        self.force_requested = threading.Event()
        self.last_signal: int | None = None
        self.termination_signal_count = 0
        self.callback_errors: list[str] = []
        self._previous_handlers: dict[int, Any] = {}
        self._installed = False

    def _callback(self, callback: Callable[[int], None] | None, signum: int) -> None:
        if callback is None:
            return
        try:
            callback(signum)
        except (
            Exception
        ) as error:  # Signal state must remain visible even if logging fails.
            self.callback_errors.append(f"{type(error).__name__}: {error}")

    def _handle(self, signum: int, frame: Any) -> None:
        del frame
        self.last_signal = signum
        if hasattr(signal, "SIGUSR1") and signum == signal.SIGUSR1:
            self.checkpoint_requested.set()
            self._callback(self.on_checkpoint, signum)
            return

        self.termination_signal_count += 1
        if self.termination_signal_count == 1:
            self.checkpoint_requested.set()
            self.stop_requested.set()
            self._callback(self.on_checkpoint, signum)
            self._callback(self.on_graceful, signum)
            return

        self.force_requested.set()
        self._callback(self.on_forced, signum)
        if self.raise_on_second:
            raise ForcedTermination(signum)

    def install(self) -> "SignalController":
        if self._installed:
            raise RuntimeError("signal controller is already installed")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "signal handlers may only be installed from the main thread"
            )
        handled = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            handled.append(signal.SIGUSR1)
        for signum in handled:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        self._installed = True
        return self

    def restore(self) -> None:
        if not self._installed:
            return
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()
        self._installed = False

    def consume_checkpoint_request(self) -> bool:
        requested = self.checkpoint_requested.is_set()
        if requested:
            self.checkpoint_requested.clear()
        return requested

    def check_forced(self) -> None:
        if self.force_requested.is_set():
            raise ForcedTermination(self.last_signal or signal.SIGTERM)

    def __enter__(self) -> "SignalController":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.restore()


def tensors_to_cpu(value: T, _memo: dict[int, Any] | None = None) -> T:
    """Recursively detach tensors and make independent CPU copies.

    Standard mappings, sequences, sets, named tuples, and dataclass instances
    retain their container type.  Object aliasing is retained where practical.
    """
    memo = {} if _memo is None else _memo
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Tensor):
        result = value.detach().to(device="cpu", copy=True)
        memo[identity] = result
        return result  # type: ignore[return-value]
    if isinstance(value, dict):
        if hasattr(value, "default_factory"):
            result_dict = type(value)(value.default_factory)  # type: ignore[attr-defined,call-arg]
        else:
            try:
                result_dict = type(value)()
            except TypeError:
                result_dict = {}
        memo[identity] = result_dict
        for key, item in value.items():
            result_dict[tensors_to_cpu(key, memo)] = tensors_to_cpu(item, memo)
        return result_dict  # type: ignore[return-value]
    if isinstance(value, Mapping):
        converted_mapping = {
            tensors_to_cpu(key, memo): tensors_to_cpu(item, memo)
            for key, item in value.items()
        }
        try:
            result_mapping = type(value)(converted_mapping)
        except TypeError:
            result_mapping = converted_mapping
        memo[identity] = result_mapping
        return result_mapping  # type: ignore[return-value]
    if isinstance(value, list):
        result_list: list[Any] = []
        memo[identity] = result_list
        result_list.extend(tensors_to_cpu(item, memo) for item in value)
        return result_list  # type: ignore[return-value]
    if isinstance(value, tuple):
        converted = [tensors_to_cpu(item, memo) for item in value]
        if hasattr(value, "_fields"):
            result_tuple = type(value)(*converted)
        else:
            result_tuple = type(value)(converted)
        memo[identity] = result_tuple
        return result_tuple  # type: ignore[return-value]
    if isinstance(value, set):
        result_set = type(value)()
        memo[identity] = result_set
        result_set.update(tensors_to_cpu(item, memo) for item in value)
        return result_set  # type: ignore[return-value]
    if isinstance(value, frozenset):
        result_frozen = type(value)(tensors_to_cpu(item, memo) for item in value)
        memo[identity] = result_frozen
        return result_frozen  # type: ignore[return-value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result_object = copy.copy(value)
        memo[identity] = result_object
        for field in dataclasses.fields(value):
            object.__setattr__(
                result_object,
                field.name,
                tensors_to_cpu(getattr(value, field.name), memo),
            )
        return result_object
    return value


def estimate_checkpoint_bytes(value: Any) -> int:
    """Conservatively estimate serialized bytes without copying GPU tensors."""
    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int, int]] = set()
    tensor_bytes = 0
    object_count = 0

    def visit(item: Any) -> None:
        nonlocal tensor_bytes, object_count
        identity = id(item)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        object_count += 1
        if isinstance(item, Tensor):
            try:
                storage = item.untyped_storage()
                key = (str(item.device), storage.data_ptr(), storage.nbytes())
                if key not in seen_storages:
                    seen_storages.add(key)
                    tensor_bytes += storage.nbytes()
            except (RuntimeError, NotImplementedError):
                tensor_bytes += item.numel() * item.element_size()
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for child in item:
                visit(child)

    visit(value)
    # Zip/pickle metadata is small for normal state dicts, but use 15% plus a
    # MiB so a checkpoint never starts with a byte-exact optimistic estimate.
    return int(tensor_bytes * 1.15) + object_count * 256 + 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    generation: int
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    payload: Any
    generation: int
    metadata: dict[str, Any]
    path: Path
    recovered_from_fallback: bool
    failures: tuple[str, ...]


class CheckpointError(RuntimeError):
    """Base class for durable checkpoint failures."""


class CheckpointLoadError(CheckpointError):
    """Raised when no valid compatible checkpoint generation can be loaded."""


class InsufficientDiskSpace(CheckpointError):
    """Raised by checkpoint preflight before serialization starts."""


ConfigMismatchHook = Callable[[str, str | None, Mapping[str, Any]], bool]


class CheckpointManager:
    """Manage immutable, checksummed PyTorch checkpoint generations."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        prefix: str = "checkpoint",
        keep_last: int = 3,
        reserve_bytes: int = 128 * 1024 * 1024,
        logger: EventLogger | Callable[..., Any] | None = None,
    ) -> None:
        if not prefix or "/" in prefix or "\\" in prefix:
            raise ValueError(
                "checkpoint prefix must be a simple non-empty filename prefix"
            )
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes cannot be negative")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.keep_last = keep_last
        self.reserve_bytes = reserve_bytes
        self.logger = logger
        self.latest_path = self.directory / "latest.json"
        self._pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.pt$")
        self._mutex = threading.RLock()

    def _emit(
        self, event: str, *, level: str = "info", critical: bool = False, **data: Any
    ) -> None:
        if self.logger is None:
            return
        if isinstance(self.logger, EventLogger):
            self.logger.log(event, level=level, critical=critical, **data)
        else:
            self.logger(event=event, level=level, critical=critical, **data)

    def _paths(self, generation: int) -> tuple[Path, Path]:
        if generation < 0:
            raise ValueError("checkpoint generation cannot be negative")
        stem = f"{self.prefix}-{generation:08d}"
        return self.directory / f"{stem}.pt", self.directory / f"{stem}.meta.json"

    def _generation_numbers(self) -> list[int]:
        generations: set[int] = set()
        for path in self.directory.glob(f"{self.prefix}-*.pt"):
            match = self._pattern.match(path.name)
            if match:
                generations.add(int(match.group(1)))
        for path in self.directory.glob(f"{self.prefix}-*.meta.json"):
            name = path.name.removesuffix(".meta.json") + ".pt"
            match = self._pattern.match(name)
            if match:
                generations.add(int(match.group(1)))
        return sorted(generations)

    def next_generation(self) -> int:
        generations = self._generation_numbers()
        return (generations[-1] + 1) if generations else 1

    def preflight(self, payload: Any) -> dict[str, int]:
        """Check free space before any tensor is copied or serialized."""
        estimated = estimate_checkpoint_bytes(payload)
        free = shutil.disk_usage(self.directory).free
        required = estimated + self.reserve_bytes
        if free < required:
            raise InsufficientDiskSpace(
                f"checkpoint needs approximately {estimated:,} bytes plus "
                f"{self.reserve_bytes:,} bytes reserve, but only {free:,} bytes are free"
            )
        return {
            "estimated_bytes": estimated,
            "reserve_bytes": self.reserve_bytes,
            "required_bytes": required,
            "free_bytes": free,
        }

    def save(
        self,
        payload: Any,
        *,
        generation: int | None = None,
        config: Any | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointRecord:
        """Durably save a new generation without overwriting an old one."""
        with self._mutex:
            selected = self.next_generation() if generation is None else int(generation)
            data_path, metadata_path = self._paths(selected)
            if data_path.exists() or metadata_path.exists():
                raise CheckpointError(
                    f"checkpoint generation {selected} already exists"
                )
            space = self.preflight(payload)
            config_json = None if config is None else _jsonable(config)
            config_digest = None if config is None else stable_config_digest(config)
            cpu_payload = tensors_to_cpu(payload)

            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{data_path.name}.", suffix=".tmp", dir=self.directory
            )
            temporary = Path(temporary_name)
            try:
                output = os.fdopen(fd, "wb", closefd=True)
                fd = -1
                with output:
                    torch.save(cpu_payload, output)
                    output.flush()
                    os.fsync(output.fileno())
                size = temporary.stat().st_size
                digest = _sha256(temporary)
                # The exclusive run lock is the cross-process guard.  Recheck
                # immediately before replace to catch accidental API misuse.
                if data_path.exists() or metadata_path.exists():
                    raise CheckpointError(
                        f"checkpoint generation {selected} appeared concurrently"
                    )
                os.replace(temporary, data_path)
                _fsync_directory(self.directory)
            except BaseException:
                if fd >= 0:
                    os.close(fd)
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise

            metadata = {
                "schema": CHECKPOINT_SCHEMA,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "created_at_utc": utc_timestamp(),
                "generation": selected,
                "filename": data_path.name,
                "size_bytes": size,
                "sha256": digest,
                "config_digest": config_digest,
                "config": config_json,
                "extra": _jsonable(dict(extra_metadata or {})),
            }
            try:
                atomic_write_json(metadata_path, metadata)
                latest = {
                    "schema": LATEST_SCHEMA,
                    "schema_version": LATEST_SCHEMA_VERSION,
                    "updated_at_utc": utc_timestamp(),
                    "generation": selected,
                    "filename": data_path.name,
                    "metadata_filename": metadata_path.name,
                    "size_bytes": size,
                    "sha256": digest,
                    "config_digest": config_digest,
                }
                atomic_write_json(self.latest_path, latest)
            except BaseException:
                # The .pt remains an intentionally immutable orphan.  A future
                # save advances to the next generation, while fallback scanning
                # ignores this incomplete generation.
                raise

            self._rotate()
            self._emit(
                "checkpoint_saved",
                critical=True,
                generation=selected,
                path=str(data_path),
                size_bytes=size,
                sha256=digest,
                disk_preflight=space,
            )
            return CheckpointRecord(selected, data_path, metadata_path, metadata)

    def _rotate(self) -> None:
        generations = self._generation_numbers()
        obsolete = generations[: -self.keep_last]
        changed = False
        for generation in obsolete:
            data_path, metadata_path = self._paths(generation)
            for path in (data_path, metadata_path):
                try:
                    path.unlink()
                    changed = True
                except FileNotFoundError:
                    pass
                except OSError as error:
                    self._emit(
                        "checkpoint_rotation_failed",
                        level="warning",
                        generation=generation,
                        path=str(path),
                        error=str(error),
                    )
        if changed:
            _fsync_directory(self.directory)

    def _read_metadata(self, generation: int) -> tuple[Path, Path, dict[str, Any]]:
        data_path, metadata_path = self._paths(generation)
        with metadata_path.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointLoadError(
                f"unexpected metadata schema in {metadata_path.name}"
            )
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointLoadError(
                f"unsupported metadata version in {metadata_path.name}"
            )
        if metadata.get("generation") != generation:
            raise CheckpointLoadError(f"generation mismatch in {metadata_path.name}")
        if metadata.get("filename") != data_path.name:
            raise CheckpointLoadError(f"filename mismatch in {metadata_path.name}")
        return data_path, metadata_path, metadata

    def load_latest(
        self,
        *,
        map_location: Any = "cpu",
        expected_config: Any | None = None,
        mismatch_hook: ConfigMismatchHook | None = None,
    ) -> LoadedCheckpoint:
        """Load newest valid generation, scanning backward after corruption.

        A configuration mismatch is rejected unless ``mismatch_hook`` explicitly
        returns true.  The hook receives ``(expected_digest, actual_digest,
        metadata)`` and can implement a controlled migration policy.
        """
        with self._mutex:
            failures: list[str] = []
            preferred: int | None = None
            if self.latest_path.exists():
                try:
                    with self.latest_path.open("r", encoding="utf-8") as source:
                        latest = json.load(source)
                    if latest.get("schema") != LATEST_SCHEMA:
                        raise CheckpointLoadError("unexpected latest.json schema")
                    if latest.get("schema_version") != LATEST_SCHEMA_VERSION:
                        raise CheckpointLoadError("unsupported latest.json version")
                    preferred = int(latest["generation"])
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                    CheckpointLoadError,
                ) as error:
                    failures.append(f"latest.json: {type(error).__name__}: {error}")
            elif self._generation_numbers():
                failures.append("latest.json: missing")

            discovered = sorted(self._generation_numbers(), reverse=True)
            candidates: list[int] = []
            if preferred is not None:
                candidates.append(preferred)
            candidates.extend(item for item in discovered if item not in candidates)
            if not candidates:
                raise CheckpointLoadError("no checkpoint generations found")

            expected_digest = (
                None
                if expected_config is None
                else stable_config_digest(expected_config)
            )
            for generation in candidates:
                try:
                    data_path, _metadata_path, metadata = self._read_metadata(
                        generation
                    )
                    actual_digest = metadata.get("config_digest")
                    if expected_digest is not None and actual_digest != expected_digest:
                        allowed = bool(
                            mismatch_hook
                            and mismatch_hook(expected_digest, actual_digest, metadata)
                        )
                        if not allowed:
                            raise CheckpointLoadError(
                                f"config digest mismatch: expected {expected_digest}, "
                                f"found {actual_digest}"
                            )
                    size = data_path.stat().st_size
                    if size != metadata.get("size_bytes"):
                        raise CheckpointLoadError(
                            f"size mismatch: expected {metadata.get('size_bytes')}, found {size}"
                        )
                    digest = _sha256(data_path)
                    if digest != metadata.get("sha256"):
                        raise CheckpointLoadError(
                            f"SHA256 mismatch: expected {metadata.get('sha256')}, found {digest}"
                        )
                    try:
                        payload = torch.load(
                            data_path, map_location=map_location, weights_only=False
                        )
                    except TypeError:  # Compatibility with older supported PyTorch.
                        payload = torch.load(data_path, map_location=map_location)
                    recovered = bool(
                        failures or (preferred is not None and generation != preferred)
                    )
                    if recovered:
                        self._emit(
                            "checkpoint_fallback_loaded",
                            level="warning",
                            critical=True,
                            generation=generation,
                            failures=failures,
                        )
                    else:
                        self._emit(
                            "checkpoint_loaded",
                            generation=generation,
                            path=str(data_path),
                        )
                    return LoadedCheckpoint(
                        payload,
                        generation,
                        metadata,
                        data_path,
                        recovered,
                        tuple(failures),
                    )
                except Exception as error:
                    failures.append(
                        f"generation {generation}: {type(error).__name__}: {error}"
                    )

            self._emit(
                "checkpoint_load_failed",
                level="error",
                critical=True,
                failures=failures,
            )
            raise CheckpointLoadError(
                "no valid compatible checkpoint; " + " | ".join(failures)
            )


def gpu_telemetry(device: str | int | torch.device | None = None) -> dict[str, Any]:
    """Return best-effort CUDA allocation, capacity, and health telemetry."""
    result: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        return result
    try:
        selected = (
            torch.device(device)
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        if selected.type != "cuda":
            result["error"] = f"requested device is not CUDA: {selected}"
            return result
        index = (
            selected.index
            if selected.index is not None
            else torch.cuda.current_device()
        )
        selected = torch.device("cuda", index)
        free, total = torch.cuda.mem_get_info(selected)
        properties = torch.cuda.get_device_properties(selected)
        result.update(
            {
                "device": str(selected),
                "device_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "free_bytes": int(free),
                "total_bytes": int(total),
                "allocated_bytes": int(torch.cuda.memory_allocated(selected)),
                "reserved_bytes": int(torch.cuda.memory_reserved(selected)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
            }
        )
        optional_queries = {
            "utilization_percent": getattr(torch.cuda, "utilization", None),
            "temperature_c": getattr(torch.cuda, "temperature", None),
            "power_draw_mw": getattr(torch.cuda, "power_draw", None),
            "clock_rate_mhz": getattr(torch.cuda, "clock_rate", None),
        }
        for key, query in optional_queries.items():
            if callable(query):
                try:
                    result[key] = int(query(selected))
                except Exception as error:
                    result[f"{key}_error"] = str(error)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def self_test(
    tempdir: str | os.PathLike[str], device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Exercise durability, fallback, config guards, locking, and signals."""
    root = Path(tempdir)
    root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(
        prefix="snn-production-support-", dir=root
    ) as work_name:
        work = Path(work_name)
        atomic_path = work / "atomic.json"
        atomic_write_json(atomic_path, {"ok": True, "count": 2})
        checks["atomic_json"] = json.loads(atomic_path.read_text()) == {
            "ok": True,
            "count": 2,
        }

        with EventLogger(
            work / "events.jsonl", "self-test", context={"device": str(device)}
        ) as logger:
            first = logger.log("started", critical=True, value=1)
            second = logger.log("progress", value=2)
        lines = [
            json.loads(line)
            for line in (work / "events.jsonl").read_text().splitlines()
        ]
        checks["event_log"] = (
            first["sequence"] == 1
            and second["sequence"] == 2
            and len(lines) == 2
            and lines[-1]["schema"] == EVENT_SCHEMA
        )

        lock_path = work / "run.lock"
        with RunLock(lock_path, run_id=None) as run_lock:
            run_lock.set_run_id("self-test")
            owner = json.loads(lock_path.read_text())
            try:
                with RunLock(lock_path, run_id="competitor"):
                    pass
            except RunLockError:
                checks["run_lock"] = owner["run_id"] == "self-test"
            else:
                checks["run_lock"] = False

        manager = CheckpointManager(work / "checkpoints", keep_last=3, reserve_bytes=0)
        tensor = torch.arange(8, dtype=torch.float32, device=device)
        config = {"width": 4, "depth": 1}
        first_record = manager.save({"tensor": tensor, "step": 1}, config=config)
        initially_loaded = manager.load_latest(expected_config=config)
        checks["checkpoint_roundtrip"] = (
            initially_loaded.generation == first_record.generation
            and initially_loaded.payload["tensor"].device.type == "cpu"
            and torch.equal(initially_loaded.payload["tensor"], tensor.cpu())
        )

        second_record = manager.save({"tensor": tensor + 1, "step": 2}, config=config)
        with second_record.path.open("ab") as corrupt:
            corrupt.write(b"intentional-corruption")
            corrupt.flush()
            os.fsync(corrupt.fileno())
        recovered = manager.load_latest(expected_config=config)
        checks["corrupt_newest_fallback"] = (
            recovered.generation == first_record.generation
            and recovered.recovered_from_fallback
            and bool(recovered.failures)
        )

        mismatch_calls: list[tuple[str, str | None]] = []

        def reject_mismatch(
            expected: str, actual: str | None, metadata: Mapping[str, Any]
        ) -> bool:
            del metadata
            mismatch_calls.append((expected, actual))
            return False

        try:
            manager.load_latest(
                expected_config={"width": 999}, mismatch_hook=reject_mismatch
            )
        except CheckpointLoadError:
            checks["config_mismatch_guard"] = bool(mismatch_calls)
        else:
            checks["config_mismatch_guard"] = False

        controller = SignalController(raise_on_second=False)
        if hasattr(signal, "SIGUSR1"):
            controller._handle(signal.SIGUSR1, None)
            usr1_ok = (
                controller.consume_checkpoint_request()
                and not controller.stop_requested.is_set()
            )
        else:
            usr1_ok = True
        controller._handle(signal.SIGTERM, None)
        first_signal_ok = (
            controller.stop_requested.is_set()
            and controller.consume_checkpoint_request()
        )
        controller._handle(signal.SIGINT, None)
        checks["signal_state_machine"] = (
            usr1_ok and first_signal_ok and controller.force_requested.is_set()
        )

        checks["recursive_cpu_copy"] = (
            tensors_to_cpu({"nested": [tensor]})["nested"][0].device.type == "cpu"
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            "production support self-test failed: " + ", ".join(failed)
        )
    return {
        "ok": True,
        "checks": checks,
        "device": str(device),
        "timestamp_utc": utc_timestamp(),
        "telemetry": gpu_telemetry(device),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tempdir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("select --self-test")
    temporary_root = args.tempdir or tempfile.gettempdir()
    print(
        json.dumps(
            self_test(temporary_root, args.device),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    _main()
