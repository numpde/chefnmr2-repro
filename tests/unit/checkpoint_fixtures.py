"""Shared synthetic checkpoint fixtures for the unit-test categories."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

import sanitize


class CheckpointCase(unittest.TestCase):
    """Temporary prepared checkpoint bundles for binding tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="checkpoint-contract-")
        self.directory = Path(self.temporary.name)

        self.directory.chmod(0o755)

    def tearDown(self):
        for path in self.directory.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)

        self.directory.chmod(0o755)
        self.temporary.cleanup()

    @staticmethod
    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def save(path, value):
        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as stream:
            torch.save(value, stream)

        return path

    def make_bundle(self, directory=None, tag="epoch1"):
        directory = Path(directory or (self.directory / tag))

        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / "model.pt"
        config = directory / "config.yaml"
        state = OrderedDict((("weight", torch.tensor([1.0, 2.0])),))
        state._metadata = OrderedDict((("", {"version": 1}),))

        self.save(
            weights,
            {
                "weights": state,
                "metadata": {"fixture": tag},
            },
        )

        config.write_text(f"fixture:\n  name: {tag}\n")

        sanitize.sanitize(
            weights,
            self.digest(weights),
            companions={config: self.digest(config)},
        )

        return (weights, config)
