#!/usr/bin/env python3
"""Conventional-network baseline for the SNN depth benchmark (PyTorch).

Trains either a plain CNN (conv3x3-ReLU-maxpool blocks, then a linear head) or
a ReLU MLP on the same IDX .gz files the C trainer reads, with the same
pixel/255 normalization, the same Adam optimizer and batch size, and the same
measurement protocol: the online training loss is averaged over the epoch
while the weights move, and after the epoch the frozen model is scored on the
test set and on a fixed prefix of the training set (--train-eval), so the
train/test loss gap is one measurement on one model.

The MLP exists as the control that separates the two things a CNN changes at
once: with widths matched to the SNN's hidden layers, SNN-vs-MLP isolates the
neuron model, and MLP-vs-CNN isolates the convolutional prior.

No batch norm, no dropout, no augmentation, no lr schedule: the SNN trainer
has none of them, and the point is a like-for-like comparison, not a leaderboard.

CSV rows are one epoch of one run:
    tag,arch,depth,width,params,seed,epoch,train_loss,test_loss,test_acc,seconds,train_eval_loss,train_eval_acc
"""

import argparse
import gzip
import math
import os
import struct
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_idx(path, expect_magic):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != expect_magic:
            sys.exit(f"kmnist_cnn: bad idx magic in {path}")
        if expect_magic == 0x803:
            rows, cols = struct.unpack(">II", f.read(8))
            if (rows, cols) != (28, 28):
                sys.exit("kmnist_cnn: idx images are not 28x28")
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8), count


def load_set(data_dir, images, labels, device):
    x, n = read_idx(os.path.join(data_dir, images), 0x803)
    y, ny = read_idx(os.path.join(data_dir, labels), 0x801)
    if n != ny:
        sys.exit("kmnist_cnn: image/label count mismatch")
    x = x.view(n, 1, 28, 28).to(device).float().div_(255.0)
    y = y.to(device).long()
    return x, y


CNN_CHANNELS = [32, 64, 128, 256]


def make_cnn(depth):
    layers = []
    in_ch, size = 1, 28
    for d in range(depth):
        layers += [nn.Conv2d(in_ch, CNN_CHANNELS[d], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        in_ch, size = CNN_CHANNELS[d], size // 2
    layers += [nn.Flatten(), nn.Linear(in_ch * size * size, 10)]
    return nn.Sequential(*layers)


def make_mlp(depth, width):
    layers, in_dim = [nn.Flatten()], 784
    for _ in range(depth):
        layers += [nn.Linear(in_dim, width), nn.ReLU()]
        in_dim = width
    layers += [nn.Linear(in_dim, 10)]
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch])
        loss_sum += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return loss_sum / len(x), correct / len(x)


def train_one(opt, device, train_x, train_y, test_x, test_y, seed, csv):
    torch.manual_seed(seed)
    model = (make_cnn(opt.depth) if opt.arch == "cnn" else make_mlp(opt.depth, opt.width)).to(device)
    params = sum(p.numel() for p in model.parameters())
    adam = torch.optim.Adam(model.parameters(), lr=opt.lr)
    gen = torch.Generator(device="cpu").manual_seed(seed ^ 0x9E3779B9)
    n_eval = min(opt.train_eval, len(train_x)) if opt.train_eval else 0

    print(f"arch={opt.arch} depth={opt.depth} width={opt.width if opt.arch == 'mlp' else '-'} "
          f"params={params} lr={opt.lr:g} seed={seed}")
    final = {}
    for epoch in range(1, opt.epochs + 1):
        t0 = time.monotonic()
        model.train()
        order = torch.randperm(len(train_x), generator=gen).to(device)
        loss_sum = 0.0
        for i in range(0, len(order), opt.batch):
            idx = order[i:i + opt.batch]
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            adam.zero_grad(set_to_none=True)
            loss.backward()
            adam.step()
            loss_sum += loss.item() * len(idx)
        train_loss = loss_sum / len(order)
        elapsed = time.monotonic() - t0

        test_loss, test_acc = evaluate(model, test_x, test_y, 1000)
        ev_loss, ev_acc = (evaluate(model, train_x[:n_eval], train_y[:n_eval], 1000)
                           if n_eval else (math.nan, math.nan))
        print(f"  epoch {epoch:2d}  train_loss {train_loss:.4f}  test_loss {test_loss:.4f}  "
              f"test_acc {100 * test_acc:6.2f}%  {elapsed:.1f}s  eval_loss {ev_loss:.4f}  "
              f"eval_acc {100 * ev_acc:6.2f}%", flush=True)
        if csv:
            csv.write(f"{opt.tag},{opt.arch},{opt.depth},{opt.width if opt.arch == 'mlp' else 0},"
                      f"{params},{seed},{epoch},{train_loss:.6f},{test_loss:.6f},{test_acc:.6f},"
                      f"{elapsed:.3f},{ev_loss:.6f},{ev_acc:.6f}\n")
            csv.flush()
        final = {"test_acc": test_acc}
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/kmnist")
    ap.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--depth", type=int, default=2, help="conv blocks (cnn) or hidden layers (mlp)")
    ap.add_argument("--width", type=int, default=256, help="hidden width (mlp only)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-eval", type=int, default=0,
                    help="score the first N train images after each epoch (0 = off)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--csv")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opt = ap.parse_args()

    if opt.arch == "cnn" and not 1 <= opt.depth <= len(CNN_CHANNELS):
        sys.exit(f"kmnist_cnn: cnn depth must be 1..{len(CNN_CHANNELS)}")
    torch.set_num_threads(2)  # leave the CPU cores to the SNN trainer
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(opt.device)

    train_x, train_y = load_set(opt.data, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", device)
    test_x, test_y = load_set(opt.data, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", device)
    print(f"kmnist_cnn: {len(train_x)} train, {len(test_x)} test, device={device}")

    csv = None
    if opt.csv:
        fresh = not os.path.exists(opt.csv)
        csv = open(opt.csv, "a")
        if fresh:
            csv.write("tag,arch,depth,width,params,seed,epoch,train_loss,test_loss,test_acc,"
                      "seconds,train_eval_loss,train_eval_acc\n")

    for s in range(opt.seeds):
        train_one(opt, device, train_x, train_y, test_x, test_y, opt.seed0 + s, csv)
    if csv:
        csv.close()


if __name__ == "__main__":
    main()
