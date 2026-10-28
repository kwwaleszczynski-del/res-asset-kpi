# Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,docs]"     # Linux/macOS: .venv/bin/python
pytest -q --cov                 # unit tests, property tests, doctests, coverage of at least 90 %
ruff check .
mypy                            # strict type check of src/
mkdocs build --strict           # documentation into site/
```

## Tests

- **Hand-worked examples.** Every formula has a test that recomputes it from scratch on a small case (the PR
  family, the bridge bars, MTBF bounds against the chi-square quantiles, the closed-form Crow-AMSAA estimate,
  the ASTM rainflow example, the GUM budget of a capacity test, DST days of 23 and 25 hours, ...).
- **Properties** (Hypothesis): the energy bridge closes for any budget, ratios and losses, by both methods;
  availabilities stay in \([0, 1]\); rainflow counting preserves the path length of the reversals; interval
  unions and first-up attribution count every moment once; unit conversions round-trip.
- **Doctests.** The `Examples` sections of the docstrings run with the tests.
- **Parity.** `validation/parity.py` compares the library with the source projects on their data (see
  [Validation](validation.md)); it needs those projects and is run by hand, not in CI.

## Continuous integration

`.github/workflows/ci.yml` runs ruff and mypy, the tests with coverage on Python 3.11, 3.12 and 3.13, the tests
on the oldest supported versions (numpy 1.26, pandas 2.1, scipy 1.11), builds the wheel and the documentation.
`.github/workflows/docs.yml` publishes the documentation to GitHub Pages from `main`.

## Releasing

`.github/workflows/release.yml` builds the sdist and the wheel for every published GitHub release and uploads
them to PyPI with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no API token). Uploading is
switched off until the repository variable `PYPI_PUBLISH` is set to `true`. To switch it on:

1. On PyPI, add a pending trusted publisher for the project `res-asset-kpi`: owner `kwwaleszczynski-del`,
   repository `res-asset-kpi`, workflow `release.yml`, environment `pypi`.
2. In the GitHub repository, create the environment `pypi` and set the variable `PYPI_PUBLISH=true`.
3. Update the version in `src/res_asset_kpi/_version.py`, `CITATION.cff` and `CHANGELOG.md`, tag `vX.Y.Z` and
   publish a GitHub release.

For a DOI, switch the repository on in Zenodo's GitHub integration (log in to Zenodo with GitHub) **before**
publishing the release; Zenodo then archives every release and reads its metadata from `.zenodo.json`.
