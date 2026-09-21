# Technology & Architecture Evaluation: Model Datasets & Weights as OCI Artifacts

**Evaluation Topic:** AI Model Weights, Training Datasets, and Runtime Artifacts as OCI (Open Container Initiative) Images  
**Author:** Antigravity (MiOS Core Architecture Team)  
**Date:** 2026-09-21  
**Status:** APPROVED  

---

## 1. Executive Summary & Problem Statement

Historically, machine learning assets (model weights, training corpora, fine-tuning datasets, tokenizers, and chat templates) have been managed through proprietary, unversioned, or cloud-specific channels (Hugging Face Hub, AWS S3/GCS buckets, bespoke Python downloaders). This introduces significant friction into immutable, containerized operating systems and production DevOps pipelines:

1. **Non-Deterministic Deployments**: Models fetched at runtime break air-gapped environments, create cold-start latency spikes, and violate Architectural Law 12 (**BAKE-NOT-FETCH**).
2. **Disconnected Supply Chains**: Container images holding the application code are cryptographically signed, SBOM-audited, and version-pinned in container registries (Quay, GHCR, Harbor), while model weights and datasets lack unified provenance and access control.
3. **Storage Duplication & Inefficient Caching**: Standard object storage downloads pull entire multi-gigabyte files over HTTP on every container start, whereas container engines (`podman`, `containerd`, `crio`) already have high-performance, deduplicated, content-addressable layer stores (`/var/lib/containers/storage`).

**The Solution:** Packaging model weights and supervised fine-tuning (SFT) datasets as standard **OCI Artifacts** and **Bound Container Images** adhering to OCI Image Spec v1.1 and the CNCF Model Distribution standards.

---

## 2. Upstream Ecosystem Landscape & Candidate Standards

| Standard / Technology | Governing Body / Project | Primary Media Types / Manifest Spec | Core Pattern | Industry Adopters |
| :--- | :--- | :--- | :--- | :--- |
| **CNCF ModelPack / OCI v1.1 Artifacts** | CNCF Sandbox / OCI | `artifactType: application/vnd.cncf.model.manifest.v1+json`<br>`application/vnd.cncf.model.config.v1+json` | Generic OCI artifact using standard layer blobs for GGUF/SafeTensors and JSON config | Harbor, CNCF, VMware, SUSE, Red Hat |
| **KitOps (ModelKit / Kitfile)** | KitOps (CNCF Sandbox) | OCI v1.1 compliant; custom layer types for datasets, code, and weights | `Kitfile` YAML manifest packing model, datasets, and docs into a tamper-proof ModelKit | Jozu Hub, Harbor, GHCR, Docker Hub |
| **InstructLab (`ilab`) / RHEL AI** | Red Hat / IBM | OCI container layers; `docker://` protocol in CLI | Distributes taxonomy data, synthetic datasets, and GGUF/SafeTensors via OCI registries | Red Hat Enterprise Linux AI, OpenShift AI |
| **KServe Modelcars** | KServe Project | OCI v1.0/v1.1 container image mounting `/models` | Packages weights inside standard image layers; mounted directly into inference pods | Kubernetes, Kubeflow, vLLM, KubeAI |
| **Ollama Model Distribution** | Ollama Inc. | `application/vnd.ollama.image.model`<br>`application/vnd.ollama.image.template` | OCI-inspired manifest, but diverges from standard OCI distribution spec | Ollama ecosystem (not vanilla OCI compliant) |

---

## 3. Weighted Evaluation Matrix

| Criteria | Weight | CNCF ModelPack / OCI v1.1 | KitOps (ModelKit) | KServe Modelcars | Ollama Custom OCI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla OCI Spec Compliance** | 25% | **5** (Native OCI v1.1) | **5** (100% OCI spec) | **5** (Standard image) | **2** (Custom HTTP headers) |
| **Tooling Interoperability** (`podman`/`oras`) | 20% | **5** (Works with any client) | **4** (Requires `kit` or `oras`) | **5** (`podman pull` native) | **2** (Fails standard pull) |
| **Multi-Asset Separation** (Weights vs Data) | 20% | **4** (Configurable layers) | **5** (First-class in Kitfile) | **3** (Flat container layers) | **2** (Model-only focus) |
| **Integration with Bootc & Immutable OS** | 20% | **5** (Direct bind to ostree/var) | **4** (User-space unpacking) | **5** (Law 3 BOUND-IMAGES) | **3** (Daemon-specific) |
| **Supply Chain Attestation** (Cosign/Sigstore)| 15% | **5** (OCI Subject referral) | **5** (Tamper-proof digests) | **5** (Standard container cosign) | **1** (Unsupported) |
| **TOTAL WEIGHTED SCORE** | **100%** | **4.80 / 5.0** | **4.60 / 5.0** | **4.60 / 5.0** | **2.05 / 5.0** |

---

## 4. Architectural Analysis: Datasets vs. Models as OCI Artifacts

### 4.1 What Training Datasets Are Packaged as OCI Images?
In modern upstream AI DevOps, datasets are packaged as immutable OCI layer blobs to guarantee reproducibility:
1. **Synthetic SFT Corpora**:
   * E.g., MiOS 4-pillar SFT dataset (`mios-micro-sft.jsonl`) and InstructLab synthetic generation outputs (`generated_qna.jsonl`).
   * Packaged as compressed JSON Lines (`application/vnd.oci.image.layer.v1.tar+gzip` or `application/x-parquet`).
2. **Alignment & Evaluation Goldens**:
   * Two-sided evaluation suites, ground-truth schema tests, and negative-control test sets packaged alongside the model for automated CI gating.
3. **Layer Separation Pattern**:
   * Using OCI v1.1 layer descriptors, a consumer can pull *only* the dataset layer for training or *only* the weights layer for serving, without downloading both.

```
   OCI Image Manifest (application/vnd.oci.image.manifest.v1+json)
   ├── Config: application/vnd.cncf.model.config.v1+json
   ├── Layer 0: application/vnd.oci.image.layer.v1.tar (Tokenizer & Jinja Template)
   ├── Layer 1: application/vnd.cncf.model.weight.v1 (mios-micro-1.5b-q4_k_m.gguf)
   └── Layer 2: application/vnd.cncf.dataset.v1 (mios-micro-sft.jsonl)
```

### 4.2 What Model Formats Are Standardized for OCI Packaging?
1. **GGUF (Quantized Edge/CPU/Resident Inference)**:
   * Mainline standard for CPU and hybrid edge serving (`llama.cpp`, `llama-swap`).
   * Single-file binary format containing metadata, tensor shapes, and quantization tables (`Q4_K_M`, `Q8_0`).
   * Supported by MiOS-Micro (~0.99 GB footprint).
2. **SafeTensors (Full-Precision / Distributed Serving)**:
   * Zero-copy, memory-mapped format for vLLM, SGLang, and HuggingFace PyTorch pipelines.
   * Free from Python `pickle` security vulnerabilities.
3. **LoRA Adapter Checkpoints**:
   * Lightweight parameter delta checkpoints (~20–80 MB) packaged as OCI artifacts to dynamically swap task adapters on top of frozen base models.

---

## 5. Modern CI/CD Implementation Patterns

### 5.1 The Continuous Training -> Packaging -> Deployment Loop

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                            CI/CD Workflow                              │
   │           (.github/workflows/package-oci.yml / Forgejo CI)             │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                1. Distillation: mios_micro.dataset --role micro
                                       │
                2. SFT LoRA Training: mios_micro.train
                                       │
                3. GGUF Quantization: mios_micro.convert (Q4_K_M)
                                       │
                4. Two-Sided Verification: mios_micro.eval
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       OCI Artifact Packaging                           │
   │                                                                        │
   │  A. Modelcar Container Image:                                          │
   │     podman build -t ghcr.io/mios-dev/mios-micro:1.5b .                │
   │                                                                        │
   │  B. Pure Model Artifact (ORAS):                                        │
   │     oras push ghcr.io/mios-dev/mios-micro:1.5b-artifact \              │
   │       mios-micro-1.5b-q4_k_m.gguf:application/vnd.cncf.model.weight.v1│
   │       mios-micro-sft.jsonl:application/vnd.cncf.dataset.v1             │
   │                                                                        │
   │  C. Cryptographic Signature (Cosign / Sigstore):                       │
   │     cosign sign --key cosign.key ghcr.io/mios-dev/mios-micro:1.5b     │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                 MiOS Deployment & Bootc Ingestion                      │
   │                                                                        │
   │  - Law 3 (BOUND-IMAGES): Symlinked into /usr/lib/bootc/bound-images.d/ │
   │  - Law 12 (BAKE-NOT-FETCH): Pre-pulled into local storage              │
   │  - Mainline llama-swap: Served warm on :8500 with ttl: -1              │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Recommendations & MiOS Ecosystem Adoption

1. **Adopt Dual-Surface OCI Strategy**:
   * **Surface 1 (Runtime Container - Modelcar)**: Package `llama-server` + quantized GGUF weights into a minimal bootc-bound container image (`ghcr.io/mios-dev/mios-micro:1.5b`). This satisfies Architectural Law 3 (**BOUND-IMAGES**) and enables zero-dependency execution.
   * **Surface 2 (OCI Model Artifact via ORAS)**: For headless systems or dynamic LoRA fine-tuning, publish raw GGUF weights and SFT training datasets using CNCF ModelPack media types to container registries (`ghcr.io` / local `mios-forge`).
2. **Strict FOSS Grounding**:
   * Reject non-compliant research-only base models; standardize exclusively on Apache-2.0, MIT, or BSD-3 foundational weights (e.g. `Qwen2.5-Coder-1.5B-Instruct`).
3. **Supply Chain Attestation**:
   * Sign all emitted model artifacts and datasets with `cosign` to satisfy MiOS signature policies (`usr/lib/containers/policy.json`).
