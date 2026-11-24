"""Tests for the common PyTorch checkpoint preparation and admission contract."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
UPSTREAM = MODEL / "upstream"
for import_path in (MODEL, UPSTREAM):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import checkpoints


class TrainingMetadata:
    """Stand in for the inert optimizer metadata global in the MolDeTr source."""

    def __init__(self, values):
        self.values = values


class CheckpointTests(unittest.TestCase):
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

    def make_envelope(self, directory=None, tag="epoch1"):
        directory = Path(directory or self.directory / tag)
        directory.mkdir(parents=True, exist_ok=True)
        weights = directory / "model.pt"
        config = directory / "config.yaml"
        state = OrderedDict((("weight", torch.tensor([1.0, 2.0])),))
        state._metadata = OrderedDict((("", {"version": 1}),))
        self.save(
            weights,
            {
                "format": "chefnmr2-checkpoint",
                "format_version": 1,
                "kind": "inference",
                "export": {"ema_applied": True, "epoch": 1},
                "weights": state,
            },
        )
        config.write_text("dataset_args:\n  atom_decoder: [C, H]\n")
        checkpoints.sanitize(
            weights,
            self.digest(weights),
            companions={config: self.digest(config)},
        )
        return weights, config

    def mol_source_and_outputs(self):
        weights = self.directory / "moldetr.pth"
        shared = torch.tensor([1.0, 2.0], dtype=torch.float32)
        state = OrderedDict((
            ("first", shared),
            ("alias", shared),
            ("integer_buffer", torch.tensor([2], dtype=torch.int64)),
        ))
        state._metadata = OrderedDict((("", {"version": 3}),))
        source = {"model": state, "optimizer": TrainingMetadata(["discarded"])}
        self.save(weights, source)
        source_hash = self.digest(weights)

        expected = OrderedDict((
            ("first", shared.to(torch.float16)),
            ("alias", shared.to(torch.float16)),
            ("integer_buffer", torch.tensor([2], dtype=torch.int64)),
        ))
        expected._metadata = OrderedDict((("", {"version": 3}),))
        # Preserve the exact shared view that the converter is expected to retain.
        expected["alias"] = expected["first"]
        expected_file = self.directory / "expected.pth"
        self.save(expected_file, expected)
        output_hash = self.digest(expected_file)
        definitions = {
            "keep": (source_hash, "preserve"),
            "fp16": (output_hash, "float16"),
        }
        inert_global = f"{TrainingMetadata.__module__}.{TrainingMetadata.__qualname__}"
        return weights, source_hash, output_hash, definitions, inert_global

    def test_preserving_chef_envelope_records_selected_bundle_without_rewriting(self):
        weights, config = self.make_envelope()
        inode = weights.stat().st_ino
        before = self.digest(weights)

        with checkpoints.verified_weights(weights, companions=(config,)) as streams:
            loaded = torch.load(streams[weights], map_location="cpu", weights_only=True)
            self.assertEqual(loaded["format"], "chefnmr2-checkpoint")
            self.assertEqual(streams[config].read().decode(), "dataset_args:\n  atom_decoder: [C, H]\n")

        receipt = json.loads(checkpoints.marker_path(weights).read_text())
        self.assertEqual(receipt["policy"], checkpoints.POLICY)
        self.assertEqual(receipt["files"], {
            "model.pt": before,
            "config.yaml": self.digest(config),
        })
        self.assertEqual(weights.stat().st_ino, inode)

    def test_admission_checks_only_the_selected_checkpoint(self):
        selected, _ = self.make_envelope(tag="selected")
        unselected = self.directory / "not-selected" / "model.pt"
        self.save(unselected, {"weights": {"weight": torch.tensor([9.0])}})

        config = selected.with_name("config.yaml")
        with checkpoints.verified_weights(selected, companions=(config,)) as streams:
            admitted = torch.load(streams[selected], weights_only=True)
        self.assertEqual(admitted["format"], "chefnmr2-checkpoint")

        with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
            with checkpoints.verified_weights(unselected):
                self.fail("unmarked weights must not be admitted")

    def test_moldetr_extraction_cast_and_aliases_are_published_as_safe_state_dict(self):
        weights, source_hash, output_hash, definitions, inert_global = self.mol_source_and_outputs()

        checkpoints.sanitize(
            weights,
            source_hash,
            output_definitions=definitions,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,),
        )

        self.assertEqual(self.digest(weights), output_hash)
        with checkpoints.verified_weights(weights) as streams:
            state = torch.load(streams[weights], map_location="cpu", weights_only=True)
        self.assertEqual(set(state), {"first", "alias", "integer_buffer"})
        self.assertEqual(state["first"].dtype, torch.float16)
        self.assertEqual(state["integer_buffer"].dtype, torch.int64)
        self.assertEqual(state["first"].data_ptr(), state["alias"].data_ptr())
        self.assertEqual(state._metadata, OrderedDict((("", {"version": 3}),)))
        self.assertEqual(json.loads(checkpoints.marker_path(weights).read_text())["precision"], "fp16")

    def test_restricted_loader_does_not_fall_back_for_unapproved_global(self):
        weights = self.directory / "unapproved.pth"
        self.save(weights, {"metadata": TrainingMetadata(["not approved"])})
        source_hash = self.digest(weights)
        real_load = torch.load
        calls = []

        def record_load(*args, **kwargs):
            calls.append(kwargs.get("weights_only"))
            return real_load(*args, **kwargs)

        with patch.object(torch, "load", side_effect=record_load):
            with self.assertRaises(Exception):
                checkpoints.sanitize(weights, source_hash)

        self.assertTrue(calls)
        self.assertTrue(all(value is True for value in calls))
        self.assertFalse(checkpoints.marker_path(weights).exists())

    def test_failure_before_artifact_replacement_keeps_the_old_valid_receipt(self):
        weights = self.directory / "weights.pth"
        source = OrderedDict((("weight", torch.tensor([1.0], dtype=torch.float32)),))
        self.save(weights, source)
        source_hash = self.digest(weights)
        checkpoints.sanitize(
            weights,
            source_hash,
            output_definitions={"keep": (source_hash, "preserve")},
        )
        marker = checkpoints.marker_path(weights)
        old_receipt = marker.read_bytes()

        self.directory.chmod(0o755)
        expected = self.directory / "wrong-output.pth"
        self.save(expected, OrderedDict((("weight", torch.tensor([1.0], dtype=torch.float16)),)))
        wrong_hash = "0" * 64
        with self.assertRaisesRegex(ValueError, "output SHA-256"):
            checkpoints.sanitize(
                weights,
                source_hash,
                output_definitions={
                    "keep": (source_hash, "preserve"),
                    "fp16": (wrong_hash, "float16"),
                },
                precision="fp16",
            )

        self.assertEqual(marker.read_bytes(), old_receipt)
        with checkpoints.verified_weights(weights) as streams:
            admitted = torch.load(streams[weights], weights_only=True)
        self.assertEqual(admitted["weight"].dtype, torch.float32)

    def test_failure_after_replacement_leaves_no_receipt_and_retry_finishes(self):
        weights, source_hash, _, definitions, inert_global = self.mol_source_and_outputs()
        marker = checkpoints.marker_path(weights)
        original_replace = Path.replace

        def fail_receipt_replace(path, target):
            if Path(target) == marker:
                raise OSError("injected receipt publication failure")
            return original_replace(path, target)

        with patch.object(Path, "replace", fail_receipt_replace):
            with self.assertRaisesRegex(OSError, "receipt publication failure"):
                checkpoints.sanitize(
                    weights,
                    source_hash,
                    output_definitions=definitions,
                    precision="fp16",
                    extract="model",
                    inert_globals=(inert_global,),
                )

        self.assertFalse(marker.exists())
        self.assertEqual(self.digest(weights), definitions["fp16"][0])
        with self.assertRaises(FileNotFoundError):
            with checkpoints.verified_weights(weights):
                self.fail("replaced weights remain unavailable until retry publishes a receipt")

        checkpoints.sanitize(
            weights,
            source_hash,
            output_definitions=definitions,
            precision="fp16",
            extract="model",
            inert_globals=(inert_global,),
        )
        self.assertTrue(marker.exists())
        with checkpoints.verified_weights(weights) as streams:
            self.assertEqual(torch.load(streams[weights], weights_only=True)["first"].dtype, torch.float16)

    def test_invalidation_during_opening_fails_closed(self):
        weights, config = self.make_envelope()
        marker = checkpoints.marker_path(weights)
        replacement = self.directory / "replacement.pt"
        self.save(replacement, {"generation": 2})
        weights.parent.chmod(0o755)
        original_open = Path.open
        replaced = False

        def replace_after_payload_open(path, *args, **kwargs):
            nonlocal replaced
            stream = original_open(path, *args, **kwargs)
            if Path(path) == weights and not replaced:
                replaced = True
                marker.unlink()
                replacement.replace(weights)
            return stream

        with patch.object(Path, "open", replace_after_payload_open):
            with self.assertRaises((FileNotFoundError, ValueError)):
                with checkpoints.verified_weights(weights, companions=(config,)):
                    self.fail("a mixed or invalidated generation must not be yielded")
        self.assertTrue(replaced)

    def test_open_payload_descriptor_remains_on_admitted_generation_after_rename(self):
        weights = self.directory / "weights.pt"
        self.save(weights, {"generation": torch.tensor([1])})
        source_hash = self.digest(weights)
        checkpoints.sanitize(weights, source_hash)
        self.directory.chmod(0o755)
        replacement = self.directory / "replacement.pt"
        self.save(replacement, {"generation": torch.tensor([2])})

        with checkpoints.verified_weights(weights) as streams:
            replacement.replace(weights)
            admitted = torch.load(streams[weights], map_location="cpu", weights_only=True)

        current = torch.load(weights, map_location="cpu", weights_only=True)
        self.assertEqual(admitted["generation"].item(), 1)
        self.assertEqual(current["generation"].item(), 2)

    def test_binding_refuses_missing_marker_before_constructor_and_uses_selected_subset(self):
        import binding

        selected, config = self.make_envelope(tag="selected")
        unmarked = self.directory / "unmarked" / "model.pt"
        self.save(unmarked, {"weights": {"weight": torch.tensor([1.0])}})

        constructor_calls = []

        def constructor(members, *, device, recon_workers):
            constructor_calls.append((members, device, recon_workers))
            loaded = []
            for _tag, model_stream, config_stream in members:
                loaded.append((
                    torch.load(model_stream, map_location="cpu", weights_only=True, mmap=False),
                    config_stream.read(),
                ))
            return type("LoadedModel", (), {"recon_pool": None, "loaded": loaded})()

        with patch("chefnmr2.infer.ChefNMR2", side_effect=constructor):
            refused = binding.Inference(
                checkpoint_dirs=[str(unmarked.parent)], device="cpu", recon_workers=1,
            )
            with self.assertRaisesRegex(FileNotFoundError, "sanitization receipt is missing"):
                refused.initialize(assets=object())
            self.assertEqual(constructor_calls, [])

            selected_worker = binding.Inference(
                checkpoint_dirs=[str(selected.parent)], device="cpu", recon_workers=1,
            )
            selected_worker.initialize(assets=object())

        self.assertEqual(len(constructor_calls), 1)
        self.assertEqual(constructor_calls[0][0][0][0], selected.parent.name)
        self.assertEqual(selected_worker.model.loaded[0][0]["format"], "chefnmr2-checkpoint")
        self.assertEqual(selected_worker.model.loaded[0][1].decode(), config.read_text())


if __name__ == "__main__":
    unittest.main()
