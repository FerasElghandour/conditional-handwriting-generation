# Training and benchmark reproduction

The current reference is the [complete three-seed cGAN/DDPM comparison](results/comparison-v1/RESULTS.md): six freshly trained generators, fixed synthetic-only classifiers, per-class diagnostics, fresh FID, source fingerprints and portable inference exports. Every declared seed is reported.

## Reference commands

Use Python 3.14.3 on Mac ARM64 with `requirements-reference-macos.txt`. It is an observed pinned platform-specific closure, not a universal accelerator lock. The earlier `environment-validation.json` records the separately validated reference setup; the new campaign records its own runtime and exact source snapshot.

Acquire [canonical MNIST](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) under its original terms. Expected SHA-256: `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.

```sh
python comparison.py --data /path/to/mnist.npz --out runs/comparison
```

The protocol runs seeds 101/202/303: cGAN 50 epochs, batch 128, CPU; DDPM 40 epochs, batch 512, MPS. Every epoch uses 59,904 training rows. Sampling uses 10,000 balanced labels, MPS batches of 128 and explicit chunk seeds. Each classifier trains for 20 epochs only on synthetic data; its seed matches between generator pairs. See `COMPARISON_PROTOCOL.json`. The generic `train-diffusion` profile is a separate CPU/batch-128/seed-42 configuration; it is not the recorded MPS comparison budget.

Direct single-run examples (different seeds from the paired campaign):

```sh
python train.py --data mnist.npz --model cgan --epochs 50 --batch-size 128 --seed 42 --device cpu --output runs/cgan
python train.py --data mnist.npz --model diffusion --epochs 40 --batch-size 512 --seed 42 --device mps --output runs/diffusion
python sample.py --model cgan --checkpoint runs/cgan/cgan_generator_epochfinal.pt --samples 10000 --seed 42 --output runs/cgan-samples
python evaluate.py --data mnist.npz --samples runs/cgan-samples/samples.npz --epochs 20 --seed 2026 --output runs/utility
```

Use `comparison.py` for the recorded batched sampling route; `sample.py` is a separate direct convenience path. An existing campaign refuses source/protocol changes. Incomplete training requires explicit continuation in another folder; no partial run is silently reported as complete.

## Secondary FID reference

In a separate Python 3.11.15 Mac ARM64 environment install `requirements-fid-reference-macos.txt`, then run:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 python fid_comparison.py --data mnist.npz --campaign runs/comparison --out runs/fid
```

This explicitly permits CPU fallback for antialiased resize, qualifies eight CPU/MPS feature vectors before scoring, reuses one fixed real reference and records feature-weight hashes. `--weights` can supply the official Torch-Fidelity Inception file locally. Do not compare the new 5,000/5,000-image FID values directly with the archived experiment as if the protocols were identical. Natural RGB Inception features have limitations on MNIST, and n=3 is descriptive.

## Continue training

The `resume-cgan` profile and `train.py --resume` retain models, optimizers, completed epochs, history and random/data-loader state under an identical configuration/runtime contract. `--epochs` is the total target, and the output directory must be new. Model-only inference exports cannot resume. [MPS_REPRODUCIBILITY.md](MPS_REPRODUCIBILITY.md) documents the failed bitwise full-data MPS check; restored state does not guarantee exact accelerator continuation.

## Offline checks and profiles

```sh
python run_reproduction.py --profile checks --output runs/checks
```

`reproduction.json` declares exact commands for checks, generic single-model training, the earlier fresh-cGAN evaluation, CPU continuation and the current paired comparison. The checks are small numerical fixtures, not MNIST performance benchmarks. The reproduction driver records source/input fingerprints and explicit outcomes.

## Earlier records retained

| Evidence | Result | Boundary |
|---|---|---|
| Earlier fresh cGAN, generator seed 42 / classifier seed 2026 | 93.28% real-test accuracy | One 50-epoch CPU generator and one classifier; `results/fresh-cgan/` |
| Recovered cGAN weights, fresh classifier | 93.03% real-test accuracy | Recovered generator; `results/recovered-cgan/` |
| Archived cGAN / DDPM study | FID 51.94 / 7.28; utility 92.97% / 97.25% | Saved original experiment |
| Archived T4 sampling, 10,000 images | 0.269 / 255.235 seconds | Historical runtime; not repeated on a T4 |

The current campaign newly completes full DDPM training for three seeds. It does not repeat the exact archived training/feature/timing configuration. The earlier portable `checkpoints/cgan_fresh50.pt` retains its separately recorded export verification; the new six exports have their own verification and source/model fingerprints.

Source acquisition and reuse boundaries remain in [DATA.md](DATA.md) and [LICENSING.md](LICENSING.md).
