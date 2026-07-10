# Kuzushiji-MNIST

Ten classes of cursive Japanese (*kuzushiji*) hiragana, 60 000 training and
10 000 test images, 28x28 greyscale. A deliberate drop-in replacement for MNIST:
same IDX magic numbers, same geometry, same filenames — `tools/mnist_bptt.c`
reads it with no code change at all, via `--data data/kmnist`.

It is committed (20.3 MB, the four original IDX files exactly as published) for
the same reason MNIST is: [`../../docs/kmnist_bptt.md`](../../docs/kmnist_bptt.md)
must be reproducible from a clean checkout with no network access.

Why this dataset: MNIST is saturated. Six surrogate gradients landed inside
0.16 percentage points of one another there, and eight seeds could not separate
the top four. KMNIST is the same shape and the same cost to train, but roughly
ten points harder, which is enough headroom for a surrogate comparison to
resolve. It is harder for a real reason — each class collapses several distinct
historical character forms, so the within-class variation is genuinely larger
rather than merely noisier.

| file | bytes | sha256 |
| --- | --- | --- |
| `train-images-idx3-ubyte.gz` | 18 165 135 | `51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4` |
| `train-labels-idx1-ubyte.gz` | 29 497 | `e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17` |
| `t10k-images-idx3-ubyte.gz` | 3 041 136 | `edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5` |
| `t10k-labels-idx1-ubyte.gz` | 5 120 | `20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c` |

Verify with `sha256sum -c` against the table above, or re-fetch from
`http://codh.rois.ac.jp/kmnist/dataset/kmnist/`.

Created by the ROIS-DS Center for Open Data in the Humanities (CODH). See
Clanuwat et al., *Deep Learning for Classical Japanese Literature*, 2018
(arXiv:1812.01718). Distributed under CC BY-SA 4.0.
