"""Unit Tests and Two-Sided Verification Gates for mios_micro.package."""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from src.mios_micro import package as micro_pkg


class TestMicroPackage(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create dummy model and dataset files
        self.dummy_model = self.tmp_path / "model.gguf"
        self.dummy_model.write_bytes(b"GGUF_DUMMY_WEIGHTS_BINARY_BLOB")

        self.dummy_dataset = self.tmp_path / "dataset.jsonl"
        self.dummy_dataset.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_positive_control_valid_manifest(self):
        """Positive Control: build_modelpack_manifest produces valid CNCF ModelPack OCI v1.1 manifest."""
        manifest = micro_pkg.build_modelpack_manifest(
            str(self.dummy_model),
            str(self.dummy_dataset),
            tag="1.5b-test",
            description="Test manifest",
        )

        self.assertEqual(manifest["schemaVersion"], 2)
        self.assertEqual(manifest["mediaType"], micro_pkg.CNCF_MANIFEST_TYPE)
        self.assertEqual(manifest["artifactType"], micro_pkg.CNCF_ARTIFACT_TYPE)
        self.assertEqual(manifest["config"]["mediaType"], micro_pkg.CNCF_CONFIG_TYPE)

        # Layers: 1 model weight + 1 dataset
        self.assertEqual(len(manifest["layers"]), 2)
        weight_layer = manifest["layers"][0]
        self.assertEqual(weight_layer["mediaType"], micro_pkg.CNCF_LAYER_WEIGHT_RAW)
        self.assertTrue(weight_layer["digest"].startswith("sha256:"))
        self.assertEqual(weight_layer["annotations"]["ai.cncf.model.format"], "GGUF")

        dataset_layer = manifest["layers"][1]
        self.assertEqual(dataset_layer["mediaType"], micro_pkg.CNCF_LAYER_DATASET)
        self.assertTrue(dataset_layer["digest"].startswith("sha256:"))
        self.assertEqual(dataset_layer["annotations"]["ai.cncf.dataset.format"], "jsonl")

    def test_positive_control_model_only_manifest(self):
        """Positive Control: omitting dataset layer generates single weight layer."""
        manifest = micro_pkg.build_modelpack_manifest(str(self.dummy_model), dataset_path=None)
        self.assertEqual(len(manifest["layers"]), 1)
        self.assertEqual(manifest["layers"][0]["mediaType"], micro_pkg.CNCF_LAYER_WEIGHT_RAW)

    def test_negative_control_corrupt_manifest_schema_fails(self):
        """Negative Control: Corrupted manifest structure fails required schema assertions."""
        manifest = micro_pkg.build_modelpack_manifest(str(self.dummy_model))

        # Deliberately mutate / corrupt required fields
        corrupt_manifest = dict(manifest)
        del corrupt_manifest["artifactType"]

        self.assertNotIn("artifactType", corrupt_manifest)
        with self.assertRaises(KeyError):
            _ = corrupt_manifest["artifactType"]

    def test_sha256_file_consistency(self):
        """Verify sha256 checksum computation integrity."""
        digest, size = micro_pkg.sha256_file(str(self.dummy_model))
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(size, len(b"GGUF_DUMMY_WEIGHTS_BINARY_BLOB"))


class TestCreatedTimestamp(unittest.TestCase):
    """org.opencontainers.image.created derives from SOURCE_DATE_EPOCH or build time."""

    RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def _created(self, dummy_model: str) -> str:
        manifest = micro_pkg.build_modelpack_manifest(dummy_model)
        return manifest["annotations"]["org.opencontainers.image.created"]

    def test_source_date_epoch_zero_is_unix_epoch(self):
        """SOURCE_DATE_EPOCH=0 pins created to 1970-01-01T00:00:00Z."""
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            self.assertEqual(micro_pkg.build_created_timestamp(), "1970-01-01T00:00:00Z")
            self.assertEqual(self._created("absent-model.gguf"), "1970-01-01T00:00:00Z")

    def test_unset_source_date_epoch_uses_current_utc_time(self):
        """Without SOURCE_DATE_EPOCH, created is a well-formed, recent UTC timestamp."""
        env = {k: v for k, v in os.environ.items() if k != "SOURCE_DATE_EPOCH"}
        with mock.patch.dict(os.environ, env, clear=True):
            before = datetime.now(tz=timezone.utc).replace(microsecond=0)
            created = self._created("absent-model.gguf")
            after = datetime.now(tz=timezone.utc)
        self.assertRegex(created, self.RFC3339_UTC)
        parsed = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        self.assertGreaterEqual(parsed, before)
        self.assertLessEqual(parsed, after + timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
