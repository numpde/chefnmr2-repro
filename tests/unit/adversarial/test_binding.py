"""Missing or altered prepared inputs must fail before model construction."""

from pathlib import Path
from unittest.mock import patch

import binding
import sanitize
from checkpoint_fixtures import CheckpointCase


class BindingAdversarialTests(CheckpointCase):
    def test_missing_receipt_refuses_before_model_constructor(self):
        (weights, _) = self.make_bundle()

        weights.parent.chmod(0o755)
        sanitize.marker_path(weights).unlink()
        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2") as constructor:
            with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
                worker.initialize(assets=None)

        constructor.assert_not_called()

    def test_changed_weight_or_config_refuses_before_model_constructor(self):
        for filename in ("model.pt", "config.yaml"):
            with self.subTest(filename=filename):
                (weights, config) = self.make_bundle(tag=filename.replace(".", "-"))
                altered = weights if filename == "model.pt" else config

                altered.chmod(0o644)
                altered.write_bytes(b"not the sanitized bytes")
                worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", recon_workers=1)

                with patch("chefnmr2.infer.ChefNMR2") as constructor:
                    with self.assertRaisesRegex(ValueError, "SHA-256 .* differs"):
                        worker.initialize(assets=None)

                constructor.assert_not_called()

    def test_receipt_replacement_while_opening_refuses_before_constructor(self):
        (weights, _) = self.make_bundle()
        marker = sanitize.marker_path(weights)
        replacement = self.directory / "replacement.json"

        replacement.write_bytes(marker.read_bytes())
        weights.parent.chmod(0o755)
        original_open = Path.open
        replaced = False

        def replace_after_payload_open(path, *args, **kwargs):
            nonlocal replaced
            stream = original_open(path, *args, **kwargs)

            if (Path(path) == weights) and (not replaced):
                replaced = True

                replacement.replace(marker)

            return stream

        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2") as constructor, patch.object(Path, "open", replace_after_payload_open):
            with self.assertRaisesRegex(ValueError, "sanitization changed while opening"):
                worker.initialize(assets=None)

        self.assertTrue(replaced)
        constructor.assert_not_called()
