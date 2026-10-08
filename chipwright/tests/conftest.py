"""Make `import chipwright` work under pytest regardless of where pytest is invoked.

The repo root (the directory that *contains* the `chipwright/` package) is put on sys.path so the
tests import the package being developed, not an installed copy. Mirrors the sys.path shim each test
file carries for standalone `python3 <file>` runs.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
