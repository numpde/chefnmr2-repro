"""Create workflow-owned run directories with one timestamp/identity format."""

from datetime import datetime, timezone
from pathlib import Path
import uuid


def create_run_directory(root: Path, kind: str) -> Path:
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    directory = root / '.work' / kind / name
    directory.mkdir(parents=True)
    directory.chmod(0o755)
    return directory
