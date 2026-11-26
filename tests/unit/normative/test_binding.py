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

        def construct(members, *, device, parameter_dtype, recon_workers):
            for (tag, model_stream, config_stream) in members:
                consumed.append((tag, torch.load(model_stream, weights_only=True), config_stream.read()))

            return SimpleNamespace(recon_pool=None)

        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", parameter_dtype="float32", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2", side_effect=construct) as constructor:
            worker.initialize(assets=None)

        constructor.assert_called_once()
        self.assertEqual(len(consumed), 1)
        (tag, state, configuration) = consumed[0]

        self.assertEqual(tag, "selected")
        self.assertEqual(state["metadata"], {"fixture": "selected"})
        self.assertTrue(torch.equal(state["weights"]["weight"], torch.tensor([1., 2.])))
        self.assertEqual(configuration, config.read_bytes())

    def test_parameter_dtype_applies_to_each_loaded_member_without_changing_files(self):
        from omegaconf import OmegaConf

        class TinyModel(torch.nn.Module):
            """Small network with floating and integer buffers to check runtime casting."""

            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.model.guidance_scale = 1.0
                self.register_buffer("scale", torch.tensor([1.25]))
                self.register_buffer("count", torch.tensor([2], dtype=torch.int64))

        cfg = OmegaConf.create({"dataset_args": {}, "diffusion_process_args": {}})
        directories = []
        hashes = {}

        for tag in ("epoch1", "epoch2"):
            directory = self.directory / tag
            weights = directory / "model.pt"
            config = directory / "config.yaml"
            self.save(weights, {
                "format": "chefnmr2-checkpoint",
                "format_version": 1,
                "kind": "inference",
                "export": {"ema_applied": True, "epoch": int(tag[-1])},
                "weights": TinyModel().state_dict(),
            })
            config.write_text(OmegaConf.to_yaml(cfg))
            sanitize.sanitize(weights, self.digest(weights), companions={config: self.digest(config)})
            directories.append(str(directory))
            hashes.update({path: self.digest(path) for path in (weights, config)})

        for dtype in ("float32", "bfloat16"):
            with self.subTest(parameter_dtype=dtype):
                worker = binding.Inference(
                    checkpoint_dirs=directories, device="cpu", parameter_dtype=dtype, recon_workers=1,
                )
                with patch("chefnmr2.model.build.build_model", side_effect=lambda cfg: TinyModel()), patch(
                    "chefnmr2.infer.api.ReconPool"
                ):
                    worker.initialize(assets=None)
                    try:
                        self.assertEqual(len(worker.model.members), 2)
                        for member in worker.model.members:
                            model = member["chef"].model
                            floats = [tensor for tensor in (*model.parameters(), *model.buffers())
                                      if tensor.is_floating_point()]
                            self.assertEqual({tensor.dtype for tensor in floats}, {getattr(torch, dtype)})
                            self.assertEqual(model.count.dtype, torch.int64)
                            self.assertEqual(model.scale.item(), 1.25)
                    finally:
                        worker.close()

        self.assertEqual({path: self.digest(path) for path in hashes}, hashes)

    def test_initializer_expands_checkpoint_globs_in_sorted_order(self):
        for tag in ("epoch2", "epoch1", "selected"):
            self.make_bundle(tag=tag)

        worker = binding.Inference(
            checkpoint_dirs=[str(self.directory / "selected"), str(self.directory / "epoch*")],
            device="cpu",
            parameter_dtype="float32",
            recon_workers=1,
        )

        with patch("chefnmr2.infer.ChefNMR2") as constructor:
            worker.initialize(assets=None)

        self.assertEqual([tag for (tag, _, _) in constructor.call_args.args[0]], ["selected", "epoch1", "epoch2"])

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

        def construct(members, *, device, parameter_dtype, recon_workers):
            weights.parent.chmod(0o755)
            replacement.replace(weights)
            consumed.append(torch.load(members[0][1], weights_only=True)["metadata"])

            return SimpleNamespace(recon_pool=None)

        worker = binding.Inference(checkpoint_dirs=[str(weights.parent)], device="cpu", parameter_dtype="float32", recon_workers=1)

        with patch("chefnmr2.infer.ChefNMR2", side_effect=construct):
            worker.initialize(assets=None)

        self.assertEqual(consumed, [{"fixture": "epoch1"}])
        self.assertEqual(torch.load(weights, weights_only=True), {"changed": True})
