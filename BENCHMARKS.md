# Reproduction and benchmark scope

A fresh **50-epoch MLP cGAN** run produces samples that train a classifier to **93.28% accuracy on 10,000 real MNIST test images**. The classifier sees only 10,000 synthetic training images and trains for 20 epochs. Original MLP cGAN and residual U-Net DDPM training/sampling paths also retain compatible recovered checkpoints.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train-cgan` | Full original MLP cGAN training on canonical MNIST: 50 epochs, seed 42, batch 128, CPU; completed fresh in this release | mnist |
| `train-diffusion` | Original residual U-Net DDPM training on canonical MNIST; 40 epochs | mnist |
| `evaluate-fresh-cgan` | Published fresh 50-epoch cGAN weights; 10,000 synthetic samples; 20-epoch synthetic-only classifier; 10,000 real test images; no generator retraining | mnist |
| `resume-cgan` | Continue a compatible CPU training-state session to a total of 50 epochs; identical data/batch/seed/runtime required | mnist, session |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

The `checks` profile also strictly loads the three recovered original architectures and checks finite conditional forward/backward paths through them.

## Evidence boundary

These original architectures are separate from the compact digit_models.py components. Retained generator weights were recovered. The cGAN now also has a complete fresh 50-epoch run; DDPM has only short fresh training checks. Historical FID was not remeasured.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

Download the [canonical MNIST NPZ](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz). Its SHA-256 is checked before training. Both paths consume training images only.

```sh
python train.py --data mnist.npz --model cgan --epochs 50 --output runs/cgan
python train.py --data mnist.npz --model diffusion --epochs 40 --output runs/diffusion
python sample.py --model cgan --checkpoint runs/cgan/cgan_generator_epochfinal.pt --samples 10000 --seed 42 --output runs/cgan-samples
python evaluate.py --data mnist.npz --samples runs/cgan-samples/samples.npz --epochs 20 --output runs/utility
python sample.py --model diffusion --checkpoint runs/diffusion/diffusion_unet_epochfinal.pt --samples 20 --seed 42 --output runs/diffusion-samples
```

The cGAN 50-epoch command has now completed on all 60,000 input training rows (59,904 rows per epoch with drop_last). Fresh DDPM training was checked for one epoch on 256 images; its full 40-epoch command has not been rerun. The 93.03% utility result used recovered cGAN weights and a fresh 20-epoch classifier. Optional `--fid` additionally requires `requirements-fid.txt`, feature-extractor weights and at least 2,049 generated examples; FID was not rerun.



## Pinned local reference installation

[requirements-reference-macos.txt](requirements-reference-macos.txt) pins the installed runtime/test dependency closure for Python 3.14.3 on macOS ARM64. It accompanies the observed environment record. A new empty Python3.14.3 macOS environment installed these pinned dependencies, passed `pip check`, and passed all eight offline repository check profiles. [The validation record](environment-validation.json) binds the source revisions, Python source hashes, versions and outcomes. This is a platform-specific reference, not a Linux/CUDA lock or a repetition of every full dataset benchmark. The GitHub checks use their separately declared Linux environment.

```sh
python3.14 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
```

Use the recorded exact Python patch version when comparing numerical results. Notebook tooling and optional FID dependencies are outside this runtime/test snapshot.

## Fresh cGAN end-to-end reference

A new 50-epoch CPU run (seed 42, batch 128) completed in 148.08 seconds summed over training epochs on an Apple M2 Max. Every epoch uses 59,904 of 60,000 training rows because the final incomplete batch is dropped. A new classifier (seed 2026) trained for 20 epochs on 10,000 synthetic samples reaches **93.28% (9,328/10,000)** on the real MNIST test set. No real test images train either model.

| Generator source | Synthetic classifier train | Real test accuracy | Scope |
|---|---:|---:|---|
| Fresh 50-epoch cGAN | 10,000 images, 20 epochs | 93.28% | Complete fresh generator and classifier training |
| Recovered cGAN weights | 10,000 images, 20 epochs | 93.03% | Recovered generator; fresh classifier |

The difference is one observed run, not evidence of statistical superiority. FID and historical T4 generation latency were not remeasured. Full DDPM training remains unreplicated.

[Protocol and source revision](results/fresh-cgan/protocol.json) · [Training history](results/fresh-cgan/training.json) · [Utility metrics and confusion matrix](results/fresh-cgan/utility.json) · [Actual preview](results/fresh-cgan/grid.png)

The portable inference checkpoint is `checkpoints/cgan_fresh50.pt`. It removes optimizer state while retaining identical model tensors and metadata; reload reproduced all 10,000 generated sample arrays exactly. Full continuation requires a training-state session, never this inference export.

```sh
python run_reproduction.py --profile evaluate-fresh-cgan --input mnist=/path/to/mnist.npz --output runs/recheck-fresh
```

## Recorded command definitions

### checks

Offline behavioral regressions; not a fresh dataset benchmark

```text
{python} smoke_test.py
{python} test_restored_models.py
{python} test_resume.py
```

### train-cgan

Full original MLP cGAN training on canonical MNIST: 50 epochs, seed 42, batch 128, CPU; completed fresh in this release

```text
{python} train.py --data {input:mnist} --model cgan --epochs 50 --output {output}/cgan --batch-size 128 --seed 42 --device cpu
```

### train-diffusion

Original residual U-Net DDPM training on canonical MNIST; 40 epochs

```text
{python} train.py --data {input:mnist} --model diffusion --epochs 40 --output {output}/diffusion --batch-size 128 --seed 42 --device cpu
```

### evaluate-fresh-cgan

Published fresh 50-epoch cGAN weights; 10,000 synthetic samples; 20-epoch synthetic-only classifier; 10,000 real test images; no generator retraining

```text
{python} sample.py --model cgan --checkpoint checkpoints/cgan_fresh50.pt --samples 10000 --seed 42 --output {output}/samples
{python} evaluate.py --data {input:mnist} --samples {output}/samples/samples.npz --epochs 20 --output {output}/utility
```

### resume-cgan

Continue a compatible CPU training-state session to a total of 50 epochs; identical data/batch/seed/runtime required

```text
{python} train.py --data {input:mnist} --model cgan --epochs 50 --batch-size 128 --seed 42 --device cpu --resume {input:session} --output {output}/continued
```
