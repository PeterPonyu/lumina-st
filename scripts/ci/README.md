# CI / Code-level Tests

This folder is reserved for lightweight, fast tests that are suitable for continuous integration (GitHub Actions, pre-commit, etc.).

In practice, most code-level tests currently live in the sibling `tests/` directory (discovered by `pytest`).

### Rules for anything placed here
- Must run in < 90 seconds on CPU
- No real spatial data files
- No heavy model training or GPU requirement
- Pure unit tests or very small integration tests (config, flow primitives, model shapes, loss finite checks, etc.)

### How they are run in CI
```bash
python -m pytest tests/ -q --tb=no
```

If you add new tiny CI-only scripts here, they can be invoked the same way.
