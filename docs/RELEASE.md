# Releasing Chipwright

The repeatable path from a clean tree to a published PyPI release. Everything up to the two `twine upload`
commands is automated and verified in CI (the `cleanroom` job); the uploads need a PyPI token.

## 1. Pre-flight (automated — CI runs this on every push)
```bash
python3 -m pytest chipwright/tests/ -q        # the suite must be green
python3 -m build                               # -> dist/chipwright-X.Y.Z-*.whl + .tar.gz
python3 -m twine check dist/*                  # metadata must PASS on both artifacts
# clean-room: install the wheel in an isolated venv with the repo OFF the path, then use it
python3 -m venv /tmp/cw_rel && /tmp/cw_rel/bin/pip install dist/*.whl
cd /tmp && /tmp/cw_rel/bin/chipwright list && /tmp/cw_rel/bin/chipwright --target htpv68,qnn2.38 resolve zipformer-enin
```
A stranger must be able to `pip install` the wheel and `list`/`resolve` with none of the author's assets.
The index ships inside the package (`chipwright/data/index.yaml`); `sshpass`/`adb` are system deps the CLI
names at run time, not pip deps.

## 2. Version
Bump in two places (kept in sync — a test guards it): `pyproject.toml` `version` and
`chipwright/__init__.py` `__version__`. Commit `chore: release X.Y.Z`. Tag: `git tag vX.Y.Z && git push --tags`.

## 3. TestPyPI first (dry run — needs a TestPyPI token)
```bash
# token from https://test.pypi.org/manage/account/token/ — never commit it
TWINE_USERNAME=__token__ TWINE_PASSWORD="pypi-<testpypi-token>" \
  python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# verify a clean install FROM TestPyPI (deps still come from real PyPI):
python3 -m venv /tmp/cw_test && /tmp/cw_test/bin/pip install \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chipwright
/tmp/cw_test/bin/chipwright list
```

## 4. PyPI (the real release — needs a PyPI token)
```bash
# token from https://pypi.org/manage/account/token/
TWINE_USERNAME=__token__ TWINE_PASSWORD="pypi-<pypi-token>" python3 -m twine upload dist/*
# confirm:
python3 -m venv /tmp/cw_pypi && /tmp/cw_pypi/bin/pip install chipwright && /tmp/cw_pypi/bin/chipwright list
```

## Notes
- `chipwright` is free on PyPI (checked). Keep the project name stable.
- The package is MIT; the spec kit stays private; each registry entry carries its source `license`.
- `run` on a hosted variant fetches `url`+`sha256` into the content-addressed cache — so hosting the CV/ASR
  artifacts (see the registry `url` entries) is what makes a stranger's `run` work, not just `resolve`.
