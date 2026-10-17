# Data and model inputs

Smoke tests generate tensors locally. Training inputs must be N×1×28×28 images, normalized to [-1,1], with labels 0–9. The original study used MNIST; images, trained weights, and cached samples are not distributed.
