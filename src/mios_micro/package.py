"""OCI ModelPack & Kitfile Packaging Module for MiOS-Micro.

Standardizes AI model weights and 4-pillar training datasets into OCI v1.1 manifests
conforming to the CNCF ModelPack (modelpack/model-spec) and KitOps specifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CNCF_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
CNCF_ARTIFACT_TYPE = "application/vnd.cncf.model.manifest.v1+json"
CNCF_CONFIG_TYPE = "application/vnd.cncf.model.config.v1+json"
CNCF_LAYER_WEIGHT_RAW = "application/vnd.cncf.model.weight.v1.raw"
CNCF_LAYER_DATASET = "application/vnd.cncf.dataset.v1"


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Compute sha256 hash and file size in bytes."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
            size += len(chunk)
    return f"sha256:{h.hexdigest()}", size


def build_modelpack_manifest(
    model_path: str,
    dataset_path: str | None = None,
    tag: str = "1.5b",
    description: str = "MiOS-Micro 1.5B resident model",
) -> dict:
    layers = []

    # 1. Model weight layer (GGUF raw binary)
    if os.path.exists(model_path):
        digest, size = sha256_file(model_path)
    else:
        # Synthetic descriptor when packaging ahead of training
        digest = f"sha256:{hashlib.sha256(model_path.encode()).hexdigest()}"
        size = 1048576000  # ~1GB estimate

    layers.append({
        "mediaType": CNCF_LAYER_WEIGHT_RAW,
        "digest": digest,
        "size": size,
        "annotations": {
            "org.opencontainers.image.title": os.path.basename(model_path),
            "ai.cncf.model.format": "GGUF",
            "ai.cncf.model.quantization": "Q4_K_M",
            "ai.cncf.model.architecture": "qwen2",
        }
    })

    # 2. Dataset layer (if provided)
    if dataset_path and os.path.exists(dataset_path):
        ds_digest, ds_size = sha256_file(dataset_path)
        layers.append({
            "mediaType": CNCF_LAYER_DATASET,
            "digest": ds_digest,
            "size": ds_size,
            "annotations": {
                "org.opencontainers.image.title": os.path.basename(dataset_path),
                "ai.cncf.dataset.format": "jsonl",
                "ai.cncf.dataset.type": "supervised-fine-tuning",
            }
        })

    config_data = {
        "architecture": "qwen2",
        "parameters": "1.54B",
        "context_length": 8192,
        "license": "Apache-2.0",
        "tag": tag,
        "description": description,
    }
    config_raw = json.dumps(config_data, indent=2).encode("utf-8")
    config_digest = f"sha256:{hashlib.sha256(config_raw).hexdigest()}"

    manifest = {
        "schemaVersion": 2,
        "mediaType": CNCF_MANIFEST_TYPE,
        "artifactType": CNCF_ARTIFACT_TYPE,
        "config": {
            "mediaType": CNCF_CONFIG_TYPE,
            "digest": config_digest,
            "size": len(config_raw),
        },
        "layers": layers,
        "annotations": {
            "org.opencontainers.image.created": "2026-09-21T00:00:00Z",
            "org.opencontainers.image.title": "mios-micro",
            "org.opencontainers.image.version": tag,
            "org.opencontainers.image.description": description,
        }
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro OCI ModelPack Manifest Generator")
    parser.add_argument("--model", default="./models/mios-micro-1.5b-q4_k_m.gguf", help="Path to model GGUF")
    parser.add_argument("--dataset", default="mios-micro-sft.jsonl", help="Path to SFT dataset")
    parser.add_argument("--out", default="manifest.json", help="Output OCI manifest path")
    parser.add_argument("--tag", default="1.5b", help="Model tag version")
    args = parser.parse_args()

    manifest = build_modelpack_manifest(args.model, args.dataset, tag=args.tag)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[mios-micro-package] Generated CNCF ModelPack OCI manifest -> {out_path}")
    print(f"  Artifact Type: {manifest['artifactType']}")
    print(f"  Layers: {len(manifest['layers'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
