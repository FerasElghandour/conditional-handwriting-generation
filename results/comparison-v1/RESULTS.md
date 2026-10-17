# Fresh model-training comparison

Three complete pipeline seed pairs (101, 202, 303), each with fresh generator training, 10,000 balanced generated images and a fresh synthetic-only classifier. Every generator consumes the canonical MNIST training split; 59,904 images per epoch are used after dropping the final partial batch. The classifier trains for 20 epochs and is tested on the fixed 10,000 real MNIST test images. No real test image is used for either training stage.

| Generator | Seed | Real-test accuracy | Fresh Inception FID |
|---|---:|---:|---:|
| cgan | 101 | 95.24% | 46.43 |
| cgan | 202 | 81.65% | 97.43 |
| cgan | 303 | 82.87% | 99.99 |
| diffusion | 101 | 96.24% | 18.11 |
| diffusion | 202 | 95.81% | 16.05 |
| diffusion | 303 | 96.52% | 17.91 |

Mean ± sample SD across the three complete pipelines: **cGAN 86.59% ± 7.52 pp**, **DDPM 96.19% ± 0.36 pp**. The paired differences, medians, per-class recalls, confusion matrices and classifier histories are in [summary.json](summary.json). These are three pipeline pairs, not six independent classifier repetitions per generator or an inferential superiority study. Generator and classifier seeds both vary across pairs.

![Every paired seed's utility and FID, shown separately](paired-results.png)

## Class-level diagnostics

| Generator | Seed | Lowest-recall digit | Real-test recall for that digit |
|---|---:|---:|---:|
| cgan | 101 | 8 | 92.51% |
| cgan | 202 | 5 | 57.51% |
| cgan | 303 | 3 | 65.94% |
| diffusion | 101 | 7 | 91.34% |
| diffusion | 202 | 5 | 91.26% |
| diffusion | 303 | 9 | 93.06% |

Full recall vectors and confusion matrices are retained, not only average accuracy. These endpoints diagnose weaknesses of each complete synthetic-to-real pipeline. They do not alone separate generator coverage from classifier optimization or establish a causal explanation; both generator and classifier seeds vary across pipeline pairs.

## Declared budgets and interpretation

cGAN uses 50 epochs, batch 128 on CPU; DDPM uses 40 epochs, batch 512 on MPS. Both generate 10,000 samples on MPS in chunks of 128. Classifier initialization and data-order seeds match within each generator pair. Final fixed-budget generator checkpoints are evaluated; no seed or epoch is selected from test accuracy or FID. [Protocol](protocol.json).

| Generator | Seed | Optimizer steps | Sum of training epoch seconds | 10,000-sample seconds |
|---|---:|---:|---:|---:|
| cgan | 101 | 23,400 | 170.6 | 2.8 |
| cgan | 202 | 23,400 | 173.5 | 2.2 |
| cgan | 303 | 23,400 | 185.2 | 2.5 |
| diffusion | 101 | 4,680 | 1214.9 | 350.6 |
| diffusion | 202 | 4,680 | 1257.0 | 366.4 |
| diffusion | 303 | 4,680 | 1190.9 | 356.3 |

These are **different generator compute budgets**, devices and optimization rules. They do not establish compute-matched efficiency. Other jobs ran concurrently on the M2 Max, so timings are execution records rather than a controlled hardware comparison. The test split and earlier results were known before this follow-up; this is not independent external validation.

## Fresh FID, separate from the archived experiment

All six models use the same first 5,000 real test images and first 5,000 balanced generated images, with one cached real reference. FID uses Torch-Fidelity Inception-v3 2048-dimensional features, MPS convolutions with CPU fallback for antialiased resize, and float64 moments/FID on CPU. Eight CPU/MPS feature vectors passed the recorded tolerance check before scoring. [FID protocol](fid-protocol.json) · [all FID values and aggregates](fid-summary.json) · [runtime and feature-weight hash](fid-manifest.json).

Natural RGB Inception features are imperfect for digits, and finite-sample FID is biased. This fresh protocol does not reproduce the archived FID or T4 latency experiment. All three seeds are reported, including weak cGAN runs. No tuning follows these final scores.

## Inspect and reproduce

Each model/seed directory contains its actual first-100 sample grid, complete generator history, sampling record, utility confusion matrix/history, portable generator checkpoint, trained classifier and all 10,000 real-test predictions. Optimizers were stripped from generator exports; every model tensor and a CPU forward fixture matched exactly. [Export verification](exports.json). Reloaded classifiers reproduced every confusion matrix and real-test prediction in the recorded CPU reference. [Classifier replay](classifier-replay.json). Full 10,000-image synthetic arrays and training-state checkpoints are retained locally, with their fingerprints in the records; they are not included here. The inference exports cannot resume training.

The [exact executed training/evaluation source](executed-source.zip) matches every source hash in [manifest.json](manifest.json). The separately archived [FID source](fid-executed-source.zip) matches its own manifest. Source credits and license notices accompany both. Root documentation was subsequently corrected to qualify GPU continuation; the recorded experiment source is immutable.

```sh
python comparison.py --data /path/to/mnist.npz --out runs/comparison
python verify_comparison.py --data /path/to/mnist.npz --results results/comparison-v1
PYTORCH_ENABLE_MPS_FALLBACK=1 python fid_comparison.py --data /path/to/mnist.npz --campaign runs/comparison --out runs/fid
```

Use the pinned Python 3.14.3 Mac reference for training/utility, and a separate Python 3.11.15 environment with `requirements-fid-reference-macos.txt` for FID. Acquire MNIST and feature weights under their original terms. Classifier replay evaluates saved models without retraining or regenerating samples; it does not repeat FID. The training commands reproduce the protocol; identical accelerator bytes are not guaranteed.

## Continuation limit measured on full data

A separate seed-202 DDPM continuation from epoch 35 to 36 restored the same configuration and captured RNG states, but all 95 model tensors differed from uninterrupted MPS training: maximum absolute difference 0.00039144. Quality impact and numerical cause were not established. The six primary generators above used uninterrupted training. [Full explanation](../../MPS_REPRODUCIBILITY.md) · [raw equality checks](mps-resume-verification.json) · [measured drift](mps-resume-drift.json).
