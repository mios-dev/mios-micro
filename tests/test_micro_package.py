"""Tests for mios_micro.package: CNCF ModelPack v0.0.7 conformance, Kitfile 1.0.0,
and the pinned-GGUF sha256 gate.

The expected values here (media types, schema bytes, Kitfile version) are taken
from the upstream standards, not from package.py, so a regression in package.py
cannot redefine what the test accepts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import jsonschema
import yaml

import src.mios_micro as micro_root
from src.mios_micro import package as micro_pkg

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

REPO = Path(__file__).resolve().parent.parent
SCHEMA = REPO / "src" / "mios_micro" / "schema" / "modelpack-config-schema.json"

# sha256 of modelpack/model-spec schema/config-schema.json at tag v0.0.7.
UPSTREAM_SCHEMA_SHA256 = "c755da7f78371519c09df7d92e74c7997184f5c73472be20d9dfb840d0ebbe0a"

# model-spec v0.0.7 docs/spec.md: the complete set of v1 layer media types.
MODELPACK_V1_LAYER_TYPES = frozenset(
    f"application/vnd.cncf.model.{kind}.v1.{fmt}"
    for kind in ("weight", "weight.config", "doc", "code", "dataset")
    for fmt in ("raw", "tar", "tar+gzip", "tar+zstd")
)
MODELPACK_CONFIG_TYPE = "application/vnd.cncf.model.config.v1+json"
MODELPACK_ARTIFACT_TYPE = "application/vnd.cncf.model.manifest.v1+json"
OCI_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"


def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_tiny_gguf(
    path: Path,
    *,
    version: int = 3,
    architecture: str = "qwen2",
    size_label: str | None = "1.5B",
    file_type: int = 15,
    tensors: tuple[tuple[str, tuple[int, ...]], ...] = (("token_embd.weight", (8, 4)), ("output_norm.weight", (8,))),
) -> None:
    """Write a minimal GGUF file (header, metadata KVs, tensor infos) per ggml docs/gguf.md."""
    kvs = [
        _gguf_str("general.architecture") + struct.pack("<I", 8) + _gguf_str(architecture),
        _gguf_str("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type),
        _gguf_str("general.quantization_version") + struct.pack("<I", 4) + struct.pack("<I", 2),
        # A string array, as tokenizer metadata is, so array parsing is exercised.
        _gguf_str("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2)
        + _gguf_str("a") + _gguf_str("b"),
        _gguf_str(f"{architecture}.context_length") + struct.pack("<I", 4) + struct.pack("<I", 32768),
    ]
    if size_label is not None:
        kvs.append(_gguf_str("general.size_label") + struct.pack("<I", 8) + _gguf_str(size_label))
    infos = b""
    offset = 0
    for name, dims in tensors:
        infos += _gguf_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", 0, offset)
        offset += 32
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs))
    path.write_bytes(header + b"".join(kvs) + infos + b"\x00" * 32)


def load_package_settings() -> dict:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["mios-micro"]["package"]


class ModelPackCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.model = self.tmp / "models" / "tiny-v3.gguf"
        self.model.parent.mkdir()
        write_tiny_gguf(self.model)
        self.dataset = self.tmp / "data" / "sft.jsonl"
        self.dataset.parent.mkdir()
        self.dataset.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")
        self.doc = self.tmp / "README.md"
        self.doc.write_text("# card\n", encoding="utf-8")
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self._cwd)
        self.addCleanup(self._tmp.cleanup)

    def build(self, **kw):
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            return micro_pkg.build_modelpack(
                "models/tiny-v3.gguf", kw.pop("dataset", "data/sft.jsonl"), kw.pop("docs", ["README.md"]), **kw
            )


class TestModelPackConfig(ModelPackCase):
    def test_vendored_schema_is_upstream_v0_0_7(self):
        digest = hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
        assert digest == UPSTREAM_SCHEMA_SHA256, f"vendored ModelPack schema differs from v0.0.7 (sha256 {digest})"

    def test_config_validates_against_vendored_schema(self):
        _, config_raw = self.build()
        config = json.loads(config_raw)
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validators.validator_for(schema)(schema).validate(config)

    def test_config_fields_come_from_gguf_header(self):
        _, config_raw = self.build()
        config = json.loads(config_raw)
        self.assertEqual(
            config["config"],
            {"format": "gguf", "architecture": "qwen2", "paramSize": "1.5B", "quantization": "Q4_K_M"},
        )
        self.assertEqual(config["descriptor"]["licenses"], ["Apache-2.0"])
        self.assertEqual(config["descriptor"]["createdAt"], "1970-01-01T00:00:00Z")
        self.assertEqual(config["modelfs"]["type"], "layers")

    def test_param_size_computed_when_size_label_absent(self):
        write_tiny_gguf(self.model, size_label=None, tensors=(("w", (1000, 1500)),))
        _, config_raw = self.build()
        self.assertEqual(json.loads(config_raw)["config"]["paramSize"], "1.5M")

    def test_schema_rejects_unknown_config_key(self):
        """Control on the validator itself: the pre-ModelPack shape must not validate."""
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        legacy = {"architecture": "qwen2", "parameters": "1.54B", "license": "Apache-2.0"}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validators.validator_for(schema)(schema).validate(legacy)

    def test_non_gguf_and_wrong_version_rejected(self):
        write_tiny_gguf(self.model, version=2)
        with self.assertRaisesRegex(micro_pkg.PackageError, "GGUF v2, expected v3"):
            self.build()
        self.model.write_bytes(b"NOTGGUF")
        with self.assertRaisesRegex(micro_pkg.PackageError, "not a GGUF file"):
            self.build()

    def test_missing_model_is_an_error_not_a_synthetic_digest(self):
        self.model.unlink()
        with self.assertRaisesRegex(micro_pkg.PackageError, "missing file"):
            self.build()


class TestModelPackManifest(ModelPackCase):
    def test_manifest_shape(self):
        manifest, config_raw = self.build()
        self.assertEqual(manifest["schemaVersion"], 2)
        self.assertEqual(manifest["mediaType"], OCI_MANIFEST_TYPE)
        self.assertEqual(manifest["artifactType"], MODELPACK_ARTIFACT_TYPE)
        self.assertEqual(manifest["config"]["mediaType"], MODELPACK_CONFIG_TYPE)
        self.assertEqual(manifest["config"]["digest"], "sha256:" + hashlib.sha256(config_raw).hexdigest())
        self.assertEqual(manifest["config"]["size"], len(config_raw))

    def test_every_layer_media_type_is_modelpack_v1(self):
        manifest, _ = self.build()
        self.assertEqual(len(manifest["layers"]), 3)
        for layer in manifest["layers"]:
            mt = layer["mediaType"]
            assert mt in MODELPACK_V1_LAYER_TYPES, f"layer mediaType {mt} is not a ModelPack v1 media type"

    def test_layer_roles_and_annotations(self):
        manifest, _ = self.build()
        roles = [layer["mediaType"] for layer in manifest["layers"]]
        self.assertEqual(
            roles,
            [
                "application/vnd.cncf.model.weight.v1.raw",
                "application/vnd.cncf.model.dataset.v1.raw",
                "application/vnd.cncf.model.doc.v1.raw",
            ],
        )
        paths = ["models/tiny-v3.gguf", "data/sft.jsonl", "README.md"]
        for layer, rel in zip(manifest["layers"], paths, strict=True):
            self.assertEqual(layer["annotations"]["org.cncf.model.filepath"], rel)
            self.assertEqual(layer["annotations"]["org.opencontainers.image.title"], Path(rel).name)
            self.assertFalse([k for k in layer["annotations"] if k.startswith("ai.cncf.")])
            digest = "sha256:" + hashlib.sha256((self.tmp / rel).read_bytes()).hexdigest()
            self.assertEqual(layer["digest"], digest)

    def test_diff_ids_match_raw_layer_digests(self):
        manifest, config_raw = self.build()
        self.assertEqual(json.loads(config_raw)["modelfs"]["diffIds"], [layer["digest"] for layer in manifest["layers"]])

    def test_subject_makes_artifact_a_referrer(self):
        subject = {"mediaType": OCI_MANIFEST_TYPE, "digest": "sha256:" + "a" * 64, "size": 1234, "extra": "x"}
        manifest, _ = self.build(subject=subject)
        self.assertEqual(manifest["subject"], {k: subject[k] for k in ("mediaType", "digest", "size")})

    def test_cli_build_writes_config_and_manifest(self):
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            rc = micro_pkg.main(["build", "--model", "models/tiny-v3.gguf", "--dataset", "data/sft.jsonl",
                                 "--doc", "README.md", "--out-dir", "out"])
        self.assertEqual(rc, 0)
        config_raw = (self.tmp / "out" / "config.json").read_bytes()
        manifest = json.loads((self.tmp / "out" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["config"]["digest"], "sha256:" + hashlib.sha256(config_raw).hexdigest())


class TestPackagingSurfaces(unittest.TestCase):
    """Kitfile, Containerfile and pyproject name the same pinned GGUF."""

    def setUp(self):
        self.settings = load_package_settings()

    def test_kitfile_manifest_version(self):
        kitfile = yaml.safe_load((REPO / "Kitfile").read_text(encoding="utf-8"))
        got = kitfile["manifestVersion"]
        assert got == "1.0.0", f'Kitfile manifestVersion must be "1.0.0" (got {got!r})'

    def test_kitfile_ai_hint_header(self):
        lines = (REPO / "Kitfile").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("# AI-hint: "))

    def test_kitfile_model_path_is_pinned_gguf_file(self):
        kitfile = yaml.safe_load((REPO / "Kitfile").read_text(encoding="utf-8"))
        self.assertEqual(kitfile["model"]["path"], f"models/{self.settings['gguf_file']}")

    def test_kitfile_docs_exist(self):
        kitfile = yaml.safe_load((REPO / "Kitfile").read_text(encoding="utf-8"))
        docs = [d["path"] for d in kitfile["docs"]]
        self.assertEqual(docs, ["README.md", "docs/dataset-card.md"])
        for doc in docs:
            self.assertTrue((REPO / doc).is_file(), doc)

    def test_containerfile_model_path_is_pinned_gguf_file(self):
        text = (REPO / "Containerfile").read_text(encoding="utf-8")
        arg = re.search(r"^ARG GGUF_FILE=(\S+)$", text, re.MULTILINE)
        self.assertIsNotNone(arg, "Containerfile lacks ARG GGUF_FILE")
        self.assertEqual(arg.group(1), self.settings["gguf_file"])
        self.assertRegex(text, r"(?m)^ENV MODEL_PATH=/models/\$\{GGUF_FILE\}$")
        self.assertRegex(text, r"(?m)^FROM ghcr\.io/ggml-org/llama\.cpp:server$")
        self.assertNotIn("LLAMA_VERSION", text)

    def test_pyproject_pin_is_well_formed(self):
        self.assertEqual(self.settings["gguf_repo"], "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF")
        self.assertRegex(self.settings["gguf_revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(self.settings["gguf_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(self.settings["gguf_file"].endswith(".gguf"))

    def test_model_card_front_matter(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        card = yaml.safe_load(text.split("---\n", 2)[1])
        self.assertEqual(card["license"], "apache-2.0")
        self.assertEqual(card["base_model"], self.settings["base_model"])
        self.assertEqual(card["base_model_relation"], "quantized")
        self.assertEqual(card["library_name"], "gguf")
        self.assertTrue(card["datasets"])


class TestSha256Gate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = Path(self._tmp.name) / "one.gguf"
        self.file.write_bytes(b"x")
        self.sha = hashlib.sha256(b"x").hexdigest()

    def test_matching_sha_passes(self):
        micro_pkg.verify_sha256(self.file, self.sha)
        self.assertEqual(micro_pkg.main(["verify-sha256", "--file", str(self.file), "--sha256", self.sha]), 0)

    def test_one_altered_hex_digit_fails(self):
        bad = ("0" if self.sha[0] != "0" else "1") + self.sha[1:]
        with self.assertRaisesRegex(micro_pkg.PackageError, "sha256 mismatch for one.gguf"):
            micro_pkg.verify_sha256(self.file, bad)
        self.assertEqual(micro_pkg.main(["verify-sha256", "--file", str(self.file), "--sha256", bad]), 1)

    def test_missing_file_fails(self):
        with self.assertRaisesRegex(micro_pkg.PackageError, "missing file absent.gguf"):
            micro_pkg.verify_sha256(Path(self._tmp.name) / "absent.gguf", self.sha)


class TestPackageExport(unittest.TestCase):
    def test_package_listed_in_all(self):
        self.assertIn("package", micro_root.__all__)
        self.assertEqual(micro_root.__all__, sorted(micro_root.__all__))


class TestCreatedTimestamp(unittest.TestCase):
    RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_source_date_epoch_zero_is_unix_epoch(self):
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "0"}):
            self.assertEqual(micro_pkg.build_created_timestamp(), "1970-01-01T00:00:00Z")

    def test_unset_source_date_epoch_uses_current_utc_time(self):
        env = {k: v for k, v in os.environ.items() if k != "SOURCE_DATE_EPOCH"}
        with mock.patch.dict(os.environ, env, clear=True):
            before = datetime.now(tz=timezone.utc).replace(microsecond=0)
            created = micro_pkg.build_created_timestamp()
            after = datetime.now(tz=timezone.utc)
        self.assertRegex(created, self.RFC3339_UTC)
        parsed = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        self.assertGreaterEqual(parsed, before)
        self.assertLessEqual(parsed, after + timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
