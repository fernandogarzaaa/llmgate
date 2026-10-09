# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint typecheck test      # ruff, mypy, pytest (fast tests)
make e2e                      # end-to-end test with real processes
make mutation                 # testteeth mutation score of the core modules
```

Guidelines:

- Keep the core (`ratelimit`, `budget`, `cost`, `cache`, `resilience`, `routing`) free of I/O and
  framework imports so it stays unit-testable and mutation-testable. `tests/unit` must stay fast:
  testteeth runs it once per mutant.
- New behaviour needs a test that fails without it. Run `make mutation` for core changes and
  look at surviving mutants before asking for review.
- Benchmarks: never edit numbers in the README by hand. Re-run `make bench` / `make bench-cache`
  and copy from `bench/results/*.json`, stating the machine and conditions.
- Commit messages: imperative mood, one logical change per commit. Update `CHANGELOG.md`.
