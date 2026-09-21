<!-- AI-hint: Canonical roadmap for mios-micro — the dedicated 1.5B resident model and OCI artifact pipeline.
     AI-related: AGENTS.md, Kitfile, Containerfile, src/mios_micro/ -->
# MiOS-Micro Roadmap

> Project roadmap for **MiOS-Micro** — the resident miniature neural layer (<250ms latency, ~1.4GB RAM resident) and OCI artifact pipeline for MiOS.

---

## 1. Vision & Architectural Milestones

```
   ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
   │  Phase 1: Foundation  │  ───► │   Phase 2: OCI Spec   │  ───► │  Phase 3: Daemon Loop │
   │  - 4-Pillar Dataset   │       │  - CNCF ModelPack     │       │  - mios-log-watcher   │
   │  - SFT LoRA Train     │       │  - KitOps Kitfile     │       │  - mios-cron-director │
   │  - GGUF Quantization  │       │  - Bound-Image Bake   │       │  - <250ms Decodes     │
   └───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

---

## 2. Workstream Roadmap

### Phase 1: Model SFT & Quantization Pipeline (Current)
* [x] **M-01: Foundational Architecture & FOSS Selection**
  * Selected `Qwen/Qwen2.5-Coder-1.5B-Instruct` under Apache-2.0.
  * Established `LEGAL.md`, `EULA.md`, and strict OpenAI API `AGENTS.md`.
* [x] **M-02: 4-Pillar Dataset Generation (`src/mios_micro/dataset.py`)**
  * Synthesizes balanced corpus: 35% CLI, 25% Log Triage, 25% Intent Router, 15% Tool Calling.
* [x] **M-03: SFT Training Harness (`src/mios_micro/train.py`)**
  * TRL `SFTTrainer` + PEFT LoRA with hardware-agnostic accelerator detection.
* [x] **M-04: GGUF Export & Quantization (`src/mios_micro/convert.py`)**
  * Emits `Q4_K_M` (~0.99 GB) and `Q8_0` (~1.65 GB) binaries.

### Phase 2: Upstream OCI Artifact & CNCF ModelPack Integration (Completed)
* [x] **M-05: CNCF ModelPack & KitOps Compliance (`T-1104`)**
  * Standardize OCI layer media types:
    * `application/vnd.cncf.model.manifest.v1+json`
    * `application/vnd.cncf.model.weight.v1.raw`
    * `application/vnd.cncf.dataset.v1`
  * Add `Kitfile` schema v1.0.0 integration for unified package/pull workflows.
* [x] **M-06: Multi-Layer ORAS Publishing Action (`T-1105`)**
  * Automate discrete layer publishing in `.github/workflows/package-oci.yml`.
  * Support selective unpacking (`kit unpack --model` or `kit unpack --dataset`).
* [x] **M-07: Bootc Bound-Image Integration (`T-1106`)**
  * Bound container image in `/usr/lib/bootc/bound-images.d/` and registered in SBOM.
  * Satisfies Architectural Law 3 (BOUND-IMAGES) and Law 12 (BAKE-NOT-FETCH).

### Phase 3: Autonomous System Deployed Telemetry (Active)
* [x] **M-08: Two-Sided Latency & Schema Conformance Gate (`T-1107`)**
  * Two-sided unit testing of resident slot probing in `tests/test_micro_eval.py`.
  * Verifies < 250 ms decode latency and zero-hallucination negative controls.
* [ ] **M-09: Live Daemon Integration**
  * Wire `mios-log-watcher.service` and `mios-cron-director.service` directly to resident micro lane.
