<!-- AI-hint: Canonical agent entry point and source of truth for mios-micro — the dedicated, sub-250ms resident miniature model (Qwen2.5-Coder-1.5B base, Apache-2.0) and OCI artifact for MiOS.
     AI-related: /usr/share/mios/mios.toml, /usr/share/mios/llamacpp/mios-llm-light.yaml, usr/libexec/mios/mios-micro-llm, usr/libexec/mios/mios-finetune, usr/libexec/mios/mios-finetune-dataset -->
# AGENTS.md

> Canonical agent entry point for `mios-micro.git` — the dedicated miniature model
> (~1.5B parameters, resident slot, <250ms decode) and OCI artifact powering local
> system classification, event triage, command synthesis, and OpenAI-compatible tool calling
> across MiOS.
>
> **Strict OpenAI API standards and patterns ONLY.** Every interface is
> OpenAI-API-compatible verb-for-verb. No vendor-native protocols, no
> proprietary side-channels, no fallback to vendor-cloud URLs, no
> vendor-specific agent / dev-tool product references in any AI file.
>
> **Parent System Repo:** <https://github.com/mios-dev/mios>

## 0. What MiOS-Micro is

MiOS-Micro is the **resident miniature neural layer of MiOS**. While heavy reasoning and multi-turn coding are handled by the main inference lanes, an operating system requires continuous, sub-250ms evaluation for background operations:

1. **`mios-log-watcher.service`**: Real-time triage of systemd journal entries and kernel alerts.
2. **`mios-cron-director.service`**: Fast condition gating on system state before triggering scheduled tasks.
3. **`prefilter` / `agent-pipe`**: Rapid capability and verb routing across the 100+ system verbs.
4. **`mios-micro-llm` CLI**: Immediate natural language shell assistant.

### Five Load-Bearing Architectural Invariants
1. **`/var` Persists by Default**: On bootc/ostree systems, `/var` is a persistent location rather than a volatile tmpfs. Model caches and training datasets live under `/var/lib/mios/finetune/`.
2. **Unified Kernel Image (UKI) vs MOK Conflation**: The boot chain is `shim -> systemd-boot -> signed UKI` where kernel command line parameters (kargs) are baked and signed into the UKI itself.
3. **Graphics Virtualization (venus vs CUDA)**: The `venus` VirtIO GPU protocol is strictly a graphics/Vulkan transport; microVM CUDA execution requires whole-device VFIO hardware passthrough.
4. **GPU Fractioning / mediated vGPU Limit**: GPU fractioning requires a physical host PF driver; driver-free hosts utilize whole-device passthrough via `vfio-pci`.
5. **The Blade owns the hardware; the MiOS image is a guest obfuscated from it**: Hardware-facing roles live on the Blade and remain unclaimable by the hosted guest plane.

## 1. Core Model Specification

* **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **License**: **Apache-2.0** (strictly FOSS compliant; avoids proprietary research-only clauses).
* **Quantization & Footprint**:
  * `Q4_K_M`: ~0.99 GB RAM/VRAM footprint.
  * `Q8_0`: ~1.65 GB RAM/VRAM footprint.
* **Context & Attention**:
  * Native Context: 32,768 tokens (configured to 8,192 in resident slot).
  * Flash-Attention & Symmetric KV Cache: `q8_0:q8_0`.
* **Runtime Serving**:
  * Managed by `llama-swap` under the `mios-llm-light` pod (:8500).
  * Configured with `ttl: -1` (always warm, zero cold-start delay).
  * Jinja template engine enabled (`--jinja`) for native OpenAI JSON function calling.

## 2. Four-Pillar Dataset Architecture

MiOS-Micro is supervised fine-tuned (SFT) over four core operational pillars:

1. **Pillar 1: System Command & CLI Synthesis (35%)**:
   Natural language administrative intent mapped to exact system commands (`bootc`, `greenboot`, `systemctl`, `journalctl`, `podman`, `miosd`).
2. **Pillar 2: Journal Log Event Triage (25%)**:
   Raw journald/kernel log lines structured into actionable triage JSON:
   `{"severity": "...", "subsystem": "...", "root_cause": "...", "actionable": true/false, "recommended_action": "..."}`.
3. **Pillar 3: Fast Intent & Capability Routing (25%)**:
   Low-latency routing of user queries to the exact SSOT verbs defined in `[verbs]` in `mios.toml`.
4. **Pillar 4: Strict OpenAI Function Calling & Schema Conformance (15%)**:
   Structured JSON generation adhering 100% to the OpenAI tool calling specification.

## 3. OCI Artifact Distribution

MiOS-Micro is distributed as an **OCI Artifact** and **Bound Container Image** (Law 3: BOUND-IMAGES):
* Container registry target: `ghcr.io/mios-dev/mios-micro:1.5b`
* Manifest format: `application/vnd.oci.image.manifest.v1+json`
* Pre-bound at build time via `/usr/lib/bootc/bound-images.d/` for air-gapped and instant boot readiness.

## 4. Operating Rules for Agents

* **Strict OpenAI Compatibility**: All completions, embeddings, and tool-call loops must resolve standard `/v1/chat/completions` shapes.
* **No Hardcoded English**: Prompts and dataset generators must derive capabilities dynamically from the live system catalog and `mios.toml`.
* **Two-Sided Gate Verification**: Every feature, classifier, and tool definition must provide positive controls (valid inputs pass) and negative controls (corrupted/hallucinated inputs fail with clear schema errors).
