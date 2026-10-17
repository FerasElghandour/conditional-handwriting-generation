# MPS continuation qualification

Full-session checkpoints retain models, optimizers, data-loader state and Python/NumPy/Torch/MPS random states. Restoring that state does not by itself establish bitwise GPU continuation.

A real-MNIST test on the M2 Max resumed the seed-202 diffusion checkpoint from epoch 35 to 36, using batch size 512 and the same Python 3.14.3 / Torch 2.14.1 configuration. It compared that continuation with epoch 36 of the still-running uninterrupted reference. Each epoch consumed 59,904 training images; no test images were involved.

| Check | Observation |
|---|---|
| Configuration and completed epoch | Identical |
| Captured random states after the next epoch | Identical |
| Model and optimizer tensors | Not bitwise identical |
| Largest absolute parameter difference | 0.00039144 |
| Largest per-tensor relative L2 difference | 0.0031947 |
| Uninterrupted epoch-36 mean noise MSE | 0.04298670 |
| Resumed epoch-36 mean noise MSE | 0.04297717 |

All 95 model tensors differed. This test does not establish that the quality impact is negligible, or identify the underlying source of numerical divergence. The published model-quality campaign uses uninterrupted training, so this diagnostic did not replace any of its generators.

The small CPU continuation tests reproduce model/optimizer/RNG state exactly in their recorded runtime. Earlier small MPS fixtures also matched; those fixtures did not predict the full-data outcome. Deterministic algorithms are requested in the trainer, but the observed full-data MPS result is the relevant limit on an exact-continuation claim.

Use the CPU reference when exact checks are required, and verify equality in the intended runtime. Treat MPS continuation as restored training state with possible numerical drift. Cross-device, cross-version and arbitrary-workload equality have not been established. Raw checks and measured differences accompany the comparison results.

The exact training-source snapshot is retained with the benchmark. Later documentation corrections do not rewrite that source or the measured result.
