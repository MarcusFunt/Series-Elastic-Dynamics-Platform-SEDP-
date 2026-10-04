# Contributing

SEDP is currently an experimental personal research project. Small, focused pull requests are welcome.

## Development rules

- Keep the nonlinear plant equations and parameter units explicit.
- Do not change benchmark constraints when comparing controllers without documenting the change.
- Preserve a plain-servo baseline for every damping benchmark.
- Treat trained policies as experiment artifacts: document the environment/reward/config used to create them.
- Keep hardware assumptions separate from measured physical values until real calibration exists.
- New safety-critical embedded behavior should default to bounded outputs and explicit rail/motor limits.

## Validation

Before submitting changes to Python code:

```bash
python -m py_compile software/python/*.py
python tests/smoke_core.py
```

For learned-control changes, also run deterministic and randomized evaluation and include the metrics.
