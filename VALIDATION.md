# Validation scope

The offline checks cover conditioned image shapes and bounds, adversarial/diffusion gradients, strict original-architecture checkpoint loading, complete unique paired-result accounting and seed/budget validation. `test_resume.py` checks exact model/optimizer/RNG state in small split CPU fixtures for both architectures. These fixtures do not establish accelerator equality or generation quality.

```sh
python run_reproduction.py --profile checks --output runs/checks
```

The [six-run comparison](results/comparison-v1/RESULTS.md) is a separate full-MNIST training and quality experiment. All portable generator exports retained exact model tensors and matched CPU forward fixtures. Saved classifier replay reproduced all 60,000 real-test predictions across the six evaluations and their confusion matrices in the recorded CPU reference. This re-evaluates saved classifiers; it does not retrain them or regenerate synthetic inputs. The exact executed source is archived and hash-checked against the campaign manifest.

The [full-data MPS continuation diagnostic](MPS_REPRODUCIBILITY.md) did **not** reproduce model or optimizer tensors bitwise despite matching configuration and captured random states. Raw failed equality checks and measured differences are published with the comparison. No generation-quality equivalence is inferred from that diagnostic.

Reference training/utility uses Python 3.14.3 / Torch 2.14.1 on Mac ARM64. Secondary FID uses a separately pinned Python 3.11.15 / Torch 2.5.1 environment, with an explicit CPU/MPS feature qualification. Repository CI uses its declared Linux runtime. Cross-version/device numerical equality is not promised.
