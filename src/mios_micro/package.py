"""CNCF ModelPack packaging for MiOS-Micro.

Builds a ModelPack v0.0.7 model artifact (config + OCI image manifest) from a
GGUF v3 weights file, an optional SFT dataset and documentation files, and
verifies a pinned upstream GGUF against its recorded sha256.

Standards:
  * ModelPack model-spec v0.0.7 -- docs/spec.md (media types), docs/config.md
    and schema/config-schema.json (vendored as schema/modelpack-config-schema.json),
    docs/annotations.md (org.cncf.model.filepath).
  * GGUF v3 -- ggml docs/gguf.md (general.architecture, general.size_label,
    general.file_type).
  * OCI Distribution Spec v1.1 -- the optional manifest ``subject`` makes the
    artifact a referrer of the runtime image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

# OCI / ModelPack v1 media types (model-spec v0.0.7 docs/spec.md).
OCI_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
MODEL_ARTIFACT_TYPE = "application/vnd.cncf.model.manifest.v1+json"
MODEL_CONFIG_TYPE = "application/vnd.cncf.model.config.v1+json"
MODEL_LAYER_WEIGHT_RAW = "application/vnd.cncf.model.weight.v1.raw"
MODEL_LAYER_DATASET_RAW = "application/vnd.cncf.model.dataset.v1.raw"
MODEL_LAYER_DOC_RAW = "application/vnd.cncf.model.doc.v1.raw"

# Layer annotations (model-spec docs/annotations.md, OCI image-spec annotations).
ANNOTATION_FILEPATH = "org.cncf.model.filepath"
ANNOTATION_TITLE = "org.opencontainers.image.title"

LICENSES = ["Apache-2.0"]

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "modelpack-config-schema.json"

# GGUF v3 (ggml docs/gguf.md).
GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSION = 3
_GGUF_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9

# general.file_type -> quantization name (llama.cpp gguf-py LlamaFileType).
GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}

# ModelPack paramSize: <count><scale-prefix>, count with at most one decimal.
PARAM_SIZE_RE = re.compile(r"^\d+(\.\d)?[QTBMKqtbmk]$")


class PackageError(Exception):
    """Raised when inputs cannot produce a conforming ModelPack artifact."""


def build_created_timestamp() -> str:
    """Return an RFC 3339 UTC timestamp, honouring ``SOURCE_DATE_EPOCH``."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if epoch:
        moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    else:
        moment = datetime.now(tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Return (``sha256:<hex>``, size in bytes) of a file."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return f"sha256:{h.hexdigest()}", size


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise PackageError unless ``path`` exists and hashes to ``expected``."""
    p = Path(path)
    if not p.is_file():
        raise PackageError(f"missing file {p.name}")
    want = expected.strip().lower().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        raise PackageError(f"recorded sha256 for {p.name} is not 64 hex digits")
    got = sha256_file(p)[0].removeprefix("sha256:")
    if got != want:
        raise PackageError(f"sha256 mismatch for {p.name}: expected {want}, got {got}")


# --------------------------------------------------------------------------- GGUF


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise PackageError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    length = _read(f, "<Q")
    data = f.read(length)
    if len(data) != length:
        raise PackageError("truncated GGUF string")
    return data.decode("utf-8")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _GGUF_SCALARS:
        return _read(f, _GGUF_SCALARS[vtype])
    if vtype == _GGUF_STRING:
        return _read_string(f)
    if vtype == _GGUF_ARRAY:
        item_type = _read(f, "<I")
        count = _read(f, "<Q")
        return [_read_value(f, item_type) for _ in range(count)]
    raise PackageError(f"unknown GGUF value type {vtype}")


def read_gguf_header(path: str | Path) -> dict[str, Any]:
    """Parse a GGUF v3 header: ``general.*`` metadata plus the total parameter count."""
    name = Path(path).name
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise PackageError(f"{name} is not a GGUF file")
        version = _read(f, "<I")
        if version != GGUF_SUPPORTED_VERSION:
            raise PackageError(f"{name} is GGUF v{version}, expected v{GGUF_SUPPORTED_VERSION}")
        tensor_count = _read(f, "<Q")
        kv_count = _read(f, "<Q")
        general: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            value = _read_value(f, _read(f, "<I"))
            if key.startswith("general."):
                general[key] = value
        params = 0
        for _ in range(tensor_count):
            _read_string(f)
            n_dims = _read(f, "<I")
            elements = 1
            for _ in range(n_dims):
                elements *= _read(f, "<Q")
            _read(f, "<I")  # ggml_type
            _read(f, "<Q")  # offset
            params += elements
    return {"version": version, "general": general, "parameter_count": params}


def format_param_size(count: int) -> str:
    """Format a parameter count as a ModelPack paramSize (e.g. 1543714304 -> '1.5B')."""
    for prefix, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if count >= scale:
            return f"{count / scale:.1f}".removesuffix(".0") + prefix
    raise PackageError(f"parameter count {count} is too small for a paramSize")


def model_config_from_gguf(header: dict[str, Any]) -> dict[str, str]:
    """Map GGUF ``general.*`` metadata to ModelPack ``config`` fields."""
    general = header["general"]
    architecture = general.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise PackageError("GGUF header lacks general.architecture")
    size_label = general.get("general.size_label")
    if isinstance(size_label, str) and PARAM_SIZE_RE.match(size_label):
        param_size = size_label
    else:
        param_size = format_param_size(header["parameter_count"])
    file_type = general.get("general.file_type")
    if file_type not in GGUF_FILE_TYPES:
        raise PackageError(f"GGUF general.file_type {file_type!r} is not a known llama.cpp file type")
    return {
        "format": "gguf",
        "architecture": architecture,
        "paramSize": param_size,
        "quantization": GGUF_FILE_TYPES[file_type],
    }


# ---------------------------------------------------------------------- ModelPack


def _layer(path: str, media_type: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise PackageError(f"missing file {path}")
    digest, size = sha256_file(path)
    rel = Path(os.path.normpath(path)).as_posix()
    return {
        "mediaType": media_type,
        "digest": digest,
        "size": size,
        "annotations": {
            ANNOTATION_FILEPATH: rel,
            ANNOTATION_TITLE: Path(rel).name,
        },
    }


def validate_config(config: dict[str, Any]) -> None:
    """Validate a ModelPack config against the vendored v0.0.7 JSON Schema."""
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls(schema, format_checker=validator_cls.FORMAT_CHECKER).validate(config)


def build_modelpack(
    model_path: str,
    dataset_path: str | None = None,
    doc_paths: list[str] | tuple[str, ...] = (),
    name: str = "mios-micro",
    version: str = "1.5b",
    subject: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Return (OCI manifest, config JSON bytes) for a ModelPack model artifact."""
    layers = [_layer(model_path, MODEL_LAYER_WEIGHT_RAW)]
    if dataset_path:
        layers.append(_layer(dataset_path, MODEL_LAYER_DATASET_RAW))
    layers.extend(_layer(doc, MODEL_LAYER_DOC_RAW) for doc in doc_paths)

    created = build_created_timestamp()
    config = {
        "descriptor": {
            "name": name,
            "version": version,
            "licenses": list(LICENSES),
            "createdAt": created,
        },
        "config": model_config_from_gguf(read_gguf_header(model_path)),
        # Every layer is an uncompressed raw blob, so its DiffID is its digest.
        "modelfs": {"type": "layers", "diffIds": [layer["digest"] for layer in layers]},
    }
    validate_config(config)
    config_raw = json.dumps(config, indent=2, sort_keys=True).encode("utf-8")

    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "mediaType": OCI_MANIFEST_TYPE,
        "artifactType": MODEL_ARTIFACT_TYPE,
        "config": {
            "mediaType": MODEL_CONFIG_TYPE,
            "digest": f"sha256:{hashlib.sha256(config_raw).hexdigest()}",
            "size": len(config_raw),
        },
        "layers": layers,
        "annotations": {
            "org.opencontainers.image.created": created,
            "org.opencontainers.image.title": name,
            "org.opencontainers.image.version": version,
            "org.opencontainers.image.licenses": " AND ".join(LICENSES),
        },
    }
    if subject is not None:
        manifest["subject"] = {k: subject[k] for k in ("mediaType", "digest", "size")}
    return manifest, config_raw


# ---------------------------------------------------------------------------- CLI


def _cmd_build(args: argparse.Namespace) -> int:
    subject = json.loads(Path(args.subject).read_text(encoding="utf-8")) if args.subject else None
    manifest, config_raw = build_modelpack(
        args.model, args.dataset, args.doc, name=args.name, version=args.version, subject=subject
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_bytes(config_raw)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[mios-micro-package] ModelPack config + manifest -> {out} ({len(manifest['layers'])} layers)")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    verify_sha256(args.file, args.sha256)
    print(f"[mios-micro-package] sha256 ok for {Path(args.file).name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro CNCF ModelPack packager")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write config.json and manifest.json for a GGUF v3 model")
    build.add_argument("--model", required=True, help="GGUF v3 weights file (relative path is recorded)")
    build.add_argument("--dataset", help="SFT dataset (JSONL)")
    build.add_argument("--doc", action="append", default=[], help="documentation file (repeatable)")
    build.add_argument("--name", default="mios-micro")
    build.add_argument("--version", default="1.5b")
    build.add_argument("--subject", help="JSON OCI descriptor of the runtime image (referrers subject)")
    build.add_argument("--out-dir", default="modelpack")
    build.set_defaults(func=_cmd_build)

    verify = sub.add_parser("verify-sha256", help="fail unless FILE hashes to SHA256")
    verify.add_argument("--file", required=True)
    verify.add_argument("--sha256", required=True)
    verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PackageError as exc:
        print(f"[mios-micro-package] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
