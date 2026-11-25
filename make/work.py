"""Disposable build and resolver scratch under tmp."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def temporary_run_directory(root: Path, kind: str):
    """Remove this run's scratch on normal exit or exception."""
    parent = root / 'tmp' / kind
    parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='run-', dir=parent) as name:
        directory = Path(name)
        directory.chmod(0o755)
        yield directory
