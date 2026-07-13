# GPU meta-optimizer for progressively larger SNNs

`tools/snn_meta_optimizer.py` is a correctness-gated research harness for a
learned optimizer.  It does not replace the repository's C/CUDA simulator; all
methods in this comparison share one PyTorch LIF recurrence so that a semantic
difference between simulators cannot become a benchmark result.

## Model

The policy is an AlphaZero-style policy/value ResNet:

| component | shape |
| --- | --- |
| observation | 32 channels x actual SNN depth x 32 sampled quantiles |
| stem | 3x3 convolution, 32 to 256 channels, GroupNorm, SiLU |
| trunk | 43 residual blocks, two 3x3 256-channel convolutions per block |
| action head | 1x1 convolutions and quantile pooling, 8 controls per SNN layer |
| value head | 1x1 convolution, masked global pooling, 32-256-1 MLP |

The default model has **50,868,617 parameters**.  The value-only parameters are
excluded from the stricter action count; **50,851,656 parameters can affect an
update**, above the requested 50 million floor.  The depth dimension is fully
convolutional rather than flattened, so policy architecture does not impose an
artificial maximum SNN depth.

The observation contains GPU sketches of weights, true surrogate gradients,
Adam momentum/variance/update, pair-STDP eligibility, previous updates,
pre/post firing rates, membrane margins, loss and loss EMA, realized prior loss
improvement, rollback state, training progress, and layer dimensions.  The
policy tower runs once per optimizee step, not once per synapse.

The action is a bounded residual around projected Adam.  It mixes seven bases:
gradient, sign-gradient, momentum, STDP eligibility, weight decay, row-normalized
gradient, and column-normalized gradient, plus a bounded Adam scale.  The action
head is zero initialized, so a fresh policy is bit-exact projected Adam.  That is
a harness control—not evidence that anything was learned.  The scientific result
is the difference between the untrained and meta-trained residual.

## Verification gate

CUDA and a BF16-capable GPU are required for the full production-path gate.  A
small diagnostic tier is useful while editing:

```bash
python3 tools/snn_meta_optimizer.py verify --device cuda:0 --tiny-only
```

It never creates a valid experiment manifest.  The mandatory gate is:

```bash
python3 tools/snn_meta_optimizer.py verify --device cuda:0
```

The full gate covers:

- seeded GPU synthetic data and a hand-computed SNN recurrence;
- float64 surrogate finite differences and pair-STDP causality;
- a ten-step oracle comparison with `torch.optim.Adam`;
- 100 bit-exact projected-Adam policy steps, including binding clips;
- bounded nonzero policy actions, finite-bound rejection, Adam fallback, and
  atomic rollback of weights, moments, and bias-correction step;
- per-layer dead/saturated firing checks and a nonsquare two-hidden-layer SNN;
- two-step truncated meta-training and all paired benchmark methods;
- non-vacuous FP32/BF16 policy parity; and
- the actual 50M production policy, activation checkpointing, BF16, fast TF32
  mode, a tiny optimizee, backward through every policy/value parameter, fused
  AdamW state allocation, CUDA residency, finite state, and an 85% VRAM ceiling.

On success it writes `build/snn_meta_verification.json`.  The manifest binds the
source hash, exact canonical check set, PyTorch/CUDA/cuDNN versions, GPU UUID,
compute capability, VRAM size, and policy configuration.  `benchmark` has no
skip flag and refuses a missing, reduced, malformed, stale, or foreign-device
manifest.  A failed verification removes the previous manifest first.

No finite test suite can prove software “100% bug free.”  This gate is the
enforceable interpretation: no result-producing command starts until all known
correctness, integration, safety, and capacity checks pass.

## Progressive benchmark

After verification:

```bash
python3 tools/snn_meta_optimizer.py benchmark --device cuda:0
```

`DEPTHxWIDTH` entries grow from a 1x4 SNN through independent width and depth
probes.  Override the ladder for a short pilot, for example:

```bash
python3 tools/snn_meta_optimizer.py benchmark --device cuda:0 \
  --sizes 1x4,1x8,1x16 --epochs 1 --meta-steps-per-size 2
```

The benchmark uses disjoint meta-training and held-out evaluation task seeds.
Each synthetic class is a union of disconnected, antipodally paired Gaussian
clusters, giving every class the same exact population mean.  Inputs are
transformed with generator constants rather than validation/test statistics.
Before any SNN result, a train-fitted linear control must remain weak and a small
MLP must solve the validation task.

Every size reports the same initialization and update budget for:

- untrained SNN;
- hidden-layer causal pair-STDP with a supervised Adam linear readout;
- surrogate-BPTT with projected Adam;
- zero-shot policy before meta-training on the new size;
- adapted policy with loss/activity/voltage guards; and
- adapted policy without the validation/fallback guard.

The policy is best described as **surrogate-BPTT/projected-Adam plus a learned
residual**, because its observation deliberately includes the true gradient.
Local-signal-only learning is already known to fail in the prior study.

Standalone BPTT and STDP are run and scalarized before the large policy is
allocated, avoiding a fake baseline VRAM bound.  Each policy result is evaluated
and released before the next method.  Timing is synchronized end-to-end wall
time; `incremental_peak_vram_bytes` is transient allocation above the state
already live at method entry.  Meta-training time and memory are reported
separately.

Stopping uses validation accuracy and the unguarded policy, so the guarded
policy does not receive an asymmetric oracle advantage over BPTT.  Test results
are reporting-only.  Two consecutive excessive optimization gaps or guard
rejection rates establish the configured quality bound.  CUDA OOM is recorded
as a hardware capacity bound and preserves earlier results; exhausting the
ladder reports a lower bound rather than pretending a failure was found.

Results go to `build/snn_meta_results.json`, with a resumable-state snapshot
after each completed size.  Snapshots contain the policy, fused outer optimizer,
CPU/CUDA RNG states, completed-size metadata, and source hash.  The current CLI
does not yet expose resume; snapshots are last-known-good recovery artifacts.

For a statistical claim, repeat the complete command with multiple `--seed`
values and separate output paths.  The default single seed is a staged pilot,
not an uncertainty estimate.

## Resumable production protocol

`tools/snn_production.py` is the result-producing production protocol.  Unlike
the progressive research harness above, it meta-trains one policy across a
declared architecture distribution, selects the best milestone using only
development tasks, freezes that policy, and then evaluates independent width
and depth ladders against untrained, STDP, and surrogate-BPTT controls.  The
default evaluation has three independent task seeds and reports aggregated
uncertainty with two-sided Student-t 95% intervals (rather than normal
intervals, which are too narrow for three samples).

The production gate includes the core verification plus durable-log,
checkpoint-corruption fallback, configuration-compatibility, GPU round-trip,
strict aggregation, and exact interrupted/resumed trajectory checks.  Run both
gates after every change to any of the three Python sources:

```bash
python3 tools/snn_meta_optimizer.py verify --device cuda:0
python3 tools/snn_production.py verify --device cuda:0
```

The second command writes `build/snn_production_verification.json` and refuses
to validate against a stale or foreign `build/snn_meta_verification.json`.
Launch a new run into an empty directory:

```bash
RUN_DIR=build/snn-production/run-001
python3 tools/snn_production.py run --device cuda:0 --run-dir "$RUN_DIR"
```

The default command is intentionally long-running.  Operational intervals can
be adjusted without changing the scientific configuration, for example:

```bash
python3 tools/snn_production.py run --device cuda:0 --run-dir "$RUN_DIR" \
  --checkpoint-every 25 --heartbeat-seconds 30 --log-every 10
```

The runner refuses a nonempty run directory unless `--resume` is present and
holds an advisory process-lifetime lock so two writers cannot share a run.  Use
these commands to inspect and continue it:

```bash
python3 tools/snn_production.py status --run-dir "$RUN_DIR"
tail -f "$RUN_DIR/events.jsonl"
python3 tools/snn_production.py run --device cuda:0 --run-dir "$RUN_DIR" --resume
```

If the original launch changed scientific flags such as sample counts,
architecture lists, seeds, step budgets, or learning rate, repeat those exact
flags on resume.  A source, core-manifest, device, or scientific-configuration
digest mismatch is rejected.  Resume verifies checkpoint size and SHA-256 and
falls back to the newest older valid generation if the latest generation is
damaged.

The run directory layout is fixed:

| path | contents |
| --- | --- |
| `config.json` | atomic source, device, verification, and scientific configuration identity |
| `production_verification.json` | immutable run-local copy of the exact source/GPU verification gate |
| `events.jsonl` | append-only structured events, heartbeats, alarms, exceptions, and checkpoint records |
| `status.json` | atomic latest status, phase, step, termination/early-stop record, alarm summary, event, and GPU telemetry snapshot |
| `run.lock` | advisory-lock owner record; file existence alone does not mean the process is live |
| `checkpoints/latest.json` | pointer to the latest general checkpoint; the latest three generations are retained |
| `checkpoints/checkpoint-NNNNNNNN.pt` | CPU-resident resumable state, policy/optimizer when needed, cursors, and RNG state |
| `checkpoints/checkpoint-NNNNNNNN.meta.json` | checkpoint size, SHA-256, configuration digest, reason, phase, and step |
| `best/best-policy-NNNNNNNN.pt` | latest development-selected policy milestone |
| `frozen/frozen-policy-NNNNNNNN.pt` | immutable policy selected before evaluation |
| `results.json` | atomic final rows, summaries, bounds, histories, selected learning rates, alarm summary, and exact training-termination record |

Critical events are flushed and synchronized to storage.  Checkpoints are
written to a temporary file, synchronized, atomically renamed, checksummed, and
published through an atomic `latest.json`.  Disk space is checked before
serialization.  The ordinary checkpoint set rotates, while best and frozen
milestones are kept separately.

Safe candidate rollbacks are expected occasionally.  The runner records every
one and continues while the configured rolling fraction and consecutive-step
limits remain healthy.  If either limit is crossed, it records a meta-training
health bound, checkpoints the exact state, freezes the best development-selected
policy, and proceeds to evaluation.  This controlled early stop is explicit in
both `status.json` and `results.json`; it is distinct from an architecture
capacity/quality bound.  Committed non-finite values, firing-rate violations,
or VRAM limits remain hard errors. Their terminal state is checkpointed and a
plain `--resume` is refused, preventing an unattended restart from continuing
past a hard safety incident.

On Unix, `SIGUSR1` requests a checkpoint at the next optimizer/evaluation safe
boundary and lets the run continue.  The first `SIGINT` or `SIGTERM` requests a
checkpoint at the next safe boundary, records `graceful_stop`, and exits with
status 130.  A second termination signal forces immediate exit, so only work
already represented by a durable checkpoint is guaranteed.  For example:

```bash
PID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
  "$RUN_DIR/run.lock")
kill -USR1 "$PID"  # checkpoint and continue
kill -TERM "$PID"  # checkpoint and stop at the next safe boundary
```

The default width and depth ladders establish evidence only through the largest
configured SNN; a result saying that no bound was reached is a lower bound, not
an estimate of the true scaling limit.  This is a staged production protocol,
not an automatic research conclusion.
The first default completion contains multiple task repeats but only one policy
initialization and one meta-training trajectory.  Review `events.jsonl` for
fallbacks, rollbacks, VRAM alarms, exceptions, and checkpoint recovery;
review the full curves and confidence intervals in `results.json`; then repeat
the complete run in new directories with independent `--policy-seed`,
`--meta-seed`, `--dev-task-seeds`, and `--eval-task-seeds` before making a
general optimizer or scaling-bound claim.

## Safety and performance

Candidate updates are transactional.  They are bounded by an Adam-relative
residual radius, a parameter-relative step radius, per-element limits, weight,
bias, and row-norm projections, finite checks, per-hidden-layer firing limits,
voltage limits, and guard-loss regression limits.  A rejected policy action uses
same-state projected Adam only if that candidate also passes; otherwise the SNN
parameters and Adam state roll back atomically.  The actual loss improvement and
fallback/rollback flags are the next policy observation and the value-head
target.

The hot path stays on CUDA.  Synthetic data, SNN state, optimizer state, policy
construction, and oracle models are GPU resident.  The policy uses BF16
autocast, block checkpointing, fused AdamW, bounded quantile sampling, TF32, and
cuDNN autotuning in benchmark mode.  SNN recurrence, moments, projections, and
losses remain FP32.

See [the earlier policy-bound study](snn_policy_optimizer_bounds.md) for the
retractions and the important warning that “an Adam-initialized policy matches
Adam” validates plumbing only; it is not learned-optimizer evidence.
