# Installation and release verification

Create a virtual environment, activate it, and run `python -m pip install .` from the repository root. `python -m pip install -r requirements.txt` uses the same dependency declarations. For development, use `python -m pip install -e .`. Figure rendering additionally requires `python -m pip install ".[plots]"`.

For GPU execution, install a PyTorch build compatible with your GPU and CUDA driver before installing this package. Confirm `torch.cuda.is_available()` returns `True`. The release test below used the default CPU wheel; it does not verify a fresh CUDA installation.

## Verified environment

On 2026-10-03, the package was built as a wheel and installed into a new Windows virtual environment with Python 3.14, without system site packages. All dependencies were resolved from the package metadata. See [resolved dependency versions](../reports/release-test-environment.txt).

- `pip check`: no broken requirements.
- All four local release packages (Large, Medium, Small, Tiny): strict weight loading and five-crop CPU inference on one image passed using the installed package in Python isolated mode.
- The Tiny prediction CLI completed successfully.
- Both distillation unit tests passed.
- Training, cache preparation, checkpoint export and VAE download CLI help commands passed.

This is an installation and inference smoke test, not a new accuracy evaluation or a full training run. Other Python versions and operating systems were not tested in this check. Model files must be downloaded separately; pass the model package root containing `detector/` and `vae/` to `--model`.
