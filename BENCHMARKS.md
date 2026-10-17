# Reproduction and benchmark scope

Complete original **MLP cGAN** and **residual U-Net DDPM** training/sampling paths are restored with strictly compatible retained checkpoints. A fresh classifier trained only on 10,000 cGAN samples reaches **93.03% on 10,000 real MNIST test images** under the declared 20-epoch utility protocol.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train-cgan` | Original MLP cGAN training on canonical MNIST; fresh 50-epoch run, not historical checkpoint evaluation | mnist |
| `train-diffusion` | Original residual U-Net DDPM training on canonical MNIST; 40 epochs | mnist |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

The `checks` profile also strictly loads the three recovered original architectures and checks finite conditional forward/backward paths through them.

## Evidence boundary

These original architectures are separate from the compact digit_models.py components. Retained generator weights were recovered; only short fresh generator-training runs were executed locally. Historical FID was not remeasured.

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

Fresh generator training was checked for one epoch on 512 cGAN / 256 diffusion images; the 50/40-epoch commands are complete training paths, not claims that those full budgets were rerun here. The 93.03% utility result used recovered cGAN weights and a fresh 20-epoch classifier. Optional `--fid` additionally requires `requirements-fid.txt`, feature-extractor weights and at least 2,049 generated examples; FID was not rerun.


## Recorded command definitions

### checks

Offline behavioral regressions; not a fresh dataset benchmark

```text
{python} smoke_test.py
{python} test_restored_models.py
{python} test_resume.py
```

### train-cgan

Original MLP cGAN training on canonical MNIST; fresh 50-epoch run, not historical checkpoint evaluation

```text
{python} train.py --data {input:mnist} --model cgan --epochs 50 --output {output}/cgan
```

### train-diffusion

Original residual U-Net DDPM training on canonical MNIST; 40 epochs

```text
{python} train.py --data {input:mnist} --model diffusion --epochs 40 --output {output}/diffusion
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.

The commands above sample freshly trained output weights. To reproduce the published recovered-weight utility protocol, instead sample `checkpoints/cgan_generator.pt` and use the same 20-epoch classifier command. Recovered and fresh weights must not be substituted without relabeling the result. Full-session resume commands and device choices are in the [README](README.md#train-stop-and-continue).
