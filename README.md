# Conditional Handwriting Generation

**Feras Elghandour** · Technical case study

Explore the quality–latency trade-off between adversarial and diffusion-based digit generation.

## Implementation

Standalone conditioned generator/discriminator components, a conditioned noise predictor, and a cosine-schedule diffusion objective.

```mermaid
flowchart LR
    A[Validated inputs] --> B[Core implementation]
    B --> C[Behavior and error checks]
    C --> D[Inspectable outcome]
```

## Evidence and results

The completed MNIST study records cGAN FID 51.94 versus diffusion FID 7.28, with synthetic-only classifier real-test accuracy of 92.97% and 97.25%. Recorded generation time for 10,000 samples is 0.269 versus 255.235 seconds on a T4. These figures describe the saved original experiment, not the newly refactored components.

This public edition contains a standalone refactor and focused tests. Saved experimental outcomes and the current component checks have separate scopes; the refactor has not been used to recertify the original benchmark.

## Run the checks

Python 3.11 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
```

See [DATA.md](DATA.md) for input contracts and [VALIDATION.md](VALIDATION.md) for the verification scope. The modules expose small APIs that can be integrated into a larger workflow.

## Practical relevance and limitations

FID uses natural-image features and is imperfect for digit images. One seed does not establish generalization or CAPTCHA resistance. The standalone component edition is not a new reproduction of the full archived training run.

No client deployment, commercial outcome, or new live-provider performance is inferred from the component tests.
