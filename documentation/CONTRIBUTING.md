# Contributing to Mesh2Tact

Thank you for helping improve Mesh2Tact. Focused bug fixes, documentation improvements, sensor configurations, reproducible evaluation additions, and well-scoped features are welcome.

## Before opening an issue

1. Search the existing issues.
2. Confirm that the problem occurs in a clean `mesh2tact` Conda environment.
3. Record the operating system, GPU, NVIDIA driver, Python version, PyTorch version, and exact command used.
4. Remove private datasets, local paths, credentials, and identifiable camera captures from logs or screenshots.

Security vulnerabilities should not be filed publicly. Follow `SECURITY.md` instead.

## Development setup

```bash
conda env create -f environment.yml
conda activate mesh2tact
python main.py
```

Keep generated datasets, checkpoints, validation outputs, temporary files, and local preferences out of commits. The repository `.gitignore` already excludes the standard locations.

## Pull requests

- Create a focused branch from the current development branch.
- Keep unrelated formatting or cleanup out of the change.
- Explain the problem, the approach, and the verification performed.
- Include before/after images for rendering or interface changes.
- State the sensor configuration, dataset split, random seed, and evaluation protocol for numerical results.
- Do not report simulated or fitted behavior as independent physical validation.
- Update the README, configuration examples, and citation metadata when the public interface changes.

By contributing, you agree that your contribution may be distributed under the repository's MIT License.
