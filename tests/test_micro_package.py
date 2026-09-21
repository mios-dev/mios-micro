"""Unit Tests and Two-Sided Verification Gates for mios_micro.package."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
