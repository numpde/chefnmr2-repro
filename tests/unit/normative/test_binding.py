"""The worker consumes actual sanitizer receipts through its model initializer."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import binding
import sanitize
from checkpoint_fixtures import CheckpointCase


class BindingNormativeTests(CheckpointCase):
    def test_initializer_loads_selected_bundle_beside_unmarked_checkpoint(self):
        (weights, config) = self.make_bundle(tag="selected")

        self.save(self.directory / "unselected" / "model.pt", {"unmarked": True})
        consumed = []

        def construct(members, *, device, recon_workers):
            for (tag, model_stream, config_stream) in members:
                consumed.append((tag, torch.load(model_stream, weights_only=True), config_stream.read()))

            return SimpleNamespace(recon_pool=None)

        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2", side_effect=construct) as constructor:
            worker.initialize(assets=None)

        constructor.assert_called_once()
        self.assertEqual(len(consumed), 1)
        (tag, state, configuration) = consumed[0]

        self.assertEqual(tag, "selected")
        self.assertEqual(state["metadata"], {"fixture": "selected"})
        self.assertTrue(torch.equal(state["weights"]["weight"], torch.tensor([1., 2.])))
        self.assertEqual(configuration, config.read_bytes())

    def test_receipt_does_not_require_opening_unselected_members(self):
        (weights, config) = self.make_bundle()
        marker = sanitize.marker_path(weights)
        receipt = json.loads(marker.read_text())
        receipt["files"]["unselected.pt"] = "0" * 64

        weights.parent.chmod(0o755)
        marker.chmod(0o644)
        marker.write_text(json.dumps(receipt))

        with binding.verified_weights(weights, companions=(config,)) as streams:
            self.assertEqual(torch.load(streams[weights], weights_only=True)["metadata"], {"fixture": "epoch1"})

    def test_initializer_loads_verified_descriptors_after_path_replacement(self):
        (weights, config) = self.make_bundle()
        replacement = self.directory / "replacement.pt"

        self.save(replacement, {"changed": True})
        consumed = []

        def construct(members, *, device, recon_workers):
            weights.parent.chmod(0o755)
            replacement.replace(weights)
            consumed.append(torch.load(members[0][1], weights_only=True)["metadata"])

            return SimpleNamespace(recon_pool=None)

        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2", side_effect=construct):
            worker.initialize(assets=None)

        self.assertEqual(consumed, [{"fixture": "epoch1"}])
        self.assertEqual(torch.load(weights, weights_only=True), {"changed": True})
