# MNIST

The four original IDX files, gzip-compressed exactly as published, 11.6 MB in
total. They are committed so that `mnist_bptt` and everything in
[`../../docs/mnist_bptt.md`](../../docs/mnist_bptt.md) are reproducible from a
clean checkout with no network access.

| file | bytes | sha256 |
| --- | --- | --- |
| `train-images-idx3-ubyte.gz` | 9 912 422 | `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609` |
| `train-labels-idx1-ubyte.gz` | 28 881 | `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c` |
| `t10k-images-idx3-ubyte.gz` | 1 648 877 | `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6` |
| `t10k-labels-idx1-ubyte.gz` | 4 542 | `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6` |

Verify with `sha256sum -c` against the table above, or re-fetch from the CVDF
mirror (`https://storage.googleapis.com/cvdf-datasets/mnist/`), which serves the
same bytes as Yann LeCun's original distribution.

60 000 training and 10 000 test images, 28x28 greyscale, 10 classes. The IDX
format is a big-endian magic number, then the dimension sizes, then raw `uint8`
pixels; `tools/mnist_bptt.c` reads the `.gz` files directly through zlib.

MNIST is a derived work of NIST Special Database 3 and 1. It is distributed by
its authors under the Creative Commons Attribution-Share Alike 3.0 license.
