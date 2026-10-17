# Validation

Run `python smoke_test.py` for the focused offline behavior checks. Fixtures are generated locally. These checks do not train a model, access private data, call a model provider, or recertify historical metrics.

## Executed checks — October 17, 2026

Conditioned image shapes, bounded generation, adversarial gradients, and diffusion gradients passed.

Passed on Python 3.14.3; the workflow repeats these checks on Python 3.12.
