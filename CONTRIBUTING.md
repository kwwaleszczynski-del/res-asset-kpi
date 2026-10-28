# Contributing

Issues and pull requests are welcome: corrections of a formula, a definition from a contract type the library
does not cover yet, or a public data set to test against.

## Ground rules

- **Formulas need a source and a test.** A new or changed indicator comes with the public description it
  follows (paper, report, public summary of a standard) and a test that checks it against a hand-worked example.
  Do not paste text from paid standards.
- **Units and time stay explicit.** Parameter names carry their unit; series are indexed by UTC interval starts;
  reporting periods are local.
- **No silent data handling.** Checks flag; corrections and exclusions are the caller's decision and appear as
  parameters.
- **Only public data** in tests, examples and documentation.

## Development setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,docs]"     # Linux/macOS: .venv/bin/python
pytest -q --cov                 # tests, doctests and coverage (at least 90 %)
ruff check .
mypy                            # strict type check of src/
mkdocs build --strict           # documentation
```

Docstrings follow the numpydoc layout with an `Examples` section; the examples run as doctests.

## Changes and releases

Record user-visible changes under "Unreleased" in `CHANGELOG.md`. Versions follow semantic versioning; before
1.0.0 a minor version may change the API, and the changelog says how.
