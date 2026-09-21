# MiOS-Micro (1.5B)

> Dedicated resident miniature model and OCI artifact for **MiOS** — the immutable bootc/OCI Fedora agentic operating system.
> Powered by `Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0), providing sub-250ms classification, journal event triage, system command synthesis, and native OpenAI-compatible tool calling.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![OCI Artifact](https://img.shields.io/badge/OCI-ghcr.io%2Fmios--dev%2Fmios--micro-green.svg)](https://github.com/mios-dev/mios-micro/pkgs/container/mios-micro)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20%2Fv1-purple.svg)](AGENTS.md)

---

## 1. Overview

In an agentic operating system, background daemons and event monitors require continuous neural classification. Calling a 12B+ model for every journal error or condition check creates latency bottlenecks and burns VRAM.

**MiOS-Micro** solves this by maintaining a permanent, always-warm resident slot (~1.4 GB RAM / VRAM) that serves:
* **`mios-log-watcher.service`**: Real-time triage of kernel and systemd journal events.
* **`mios-cron-director.service`**: Fast condition gating before executing scheduled operations.
* **`prefilter` / `agent-pipe`**: Rapid capability and verb dispatch across the 100+ MiOS verbs.
* **`mios-micro-llm` CLI**: Instant natural language command-line operations.

Unlike legacy micro-models that rely on non-standard Pythonic special tokens, MiOS-Micro is fine-tuned for **strict OpenAI JSON tool calling and structured outputs**, adhering fully to Architectural Laws 2 and 5.

---

## 2. Model Architecture & Specifications

| Property | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | State-of-the-art reasoning/coding density at 1.5B scale |
| **License** | **Apache-2.0** | Strict FOSS compliance (no proprietary research-only clauses) |
| **Parameters** | 1.54 Billion | Capable of complex JSON schemas within edge/CPU budgets |
| **Quantization** | `Q4_K_M` (~0.99 GB) / `Q8_0` (~1.65 GB) | Resident RAM footprint < 1.5 GB |
| **Context Window** | 8,192 tokens (resident) / 32,768 native | Sized for multi-line journal streams and JSON schemas |
| **Inference Latency** | **< 250 ms** (GPU / AVX-512 CPU) | Immediate response for real-time daemon loops |
| **Serving Endpoint** | OpenAI `/v1/chat/completions` | Standard wire contract (llama-server behind llama-swap) |

---

## 3. Four-Pillar Training Protocol

The training corpus is generated through self-distillation against the live system catalog (`mios-finetune-dataset --role micro`):

```
                       ┌───────────────────────────────────────┐
                       │       Live MiOS System Surface        │
                       │  - [verbs] catalog (100+ verbs)       │
                       │  - journald / /var/log/messages       │
                       │  - bootc / greenboot / podman specs   │
                       └──────────────────┬────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                       Four-Pillar SFT Dataset                               │
   ├──────────────────────────────┬──────────────────────────────────────────────┤
   │  Pillar 1: CLI Synthesis     │  Natural language intent -> exact command    │
   │  (35% of corpus)             │  (e.g., "rollback deployment" -> "bootc      │
   │                              │   rollback")                                 │
   ├──────────────────────────────┼──────────────────────────────────────────────┤
   │  Pillar 2: Log Triage        │  Raw journal lines -> Structured JSON        │
   │  (25% of corpus)             │  (severity, subsystem, root_cause, action)   │
   ├──────────────────────────────┼──────────────────────────────────────────────┤
   │  Pillar 3: Intent Routing    │  Low-latency verb routing without full       │
   │  (25% of corpus)             │  pipeline overhead                           │
   ├──────────────────────────────┼──────────────────────────────────────────────┤
   │  Pillar 4: Function Calling  │  Strict OpenAI JSON schema validation and    │
   │  (15% of corpus)             │  two-sided tool calling conformance          │
   └──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 4. OCI Artifact Distribution

MiOS-Micro is distributed as a multi-architecture **OCI Artifact** and **Bound Container Image** conforming to the CNCF Model Distribution and OCI Image specifications.

### Pulling as an OCI Container
```bash
# Run standalone micro-server on port 8500
podman run -d --name mios-micro -p 8500:8500 ghcr.io/mios-dev/mios-micro:1.5b
```

### Pulling as an OCI Model Artifact (ORAS)
```bash
oras pull ghcr.io/mios-dev/mios-micro:1.5b-gguf
# Unpacks:
#  - mios-micro-1.5b-q4_k_m.gguf
#  - tokenizer.json
#  - chat_template.json
```

### Binding into Bootc (Architectural Law 3)
In production MiOS images, the micro-container is registered under `/usr/lib/bootc/bound-images.d/mios-micro.json`, ensuring the model image is pulled during initial OS baking and updated atomically with host OS upgrades.

---

## 5. Development & Pipeline Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
# Generate 4-pillar SFT dataset from live catalog
python3 -m mios_micro.dataset --role micro --out /tmp/mios-micro-sft.jsonl
```

### 3. Fine-Tune Model
```bash
# Run LoRA fine-tuning using HuggingFace TRL SFTTrainer
python3 -m mios_micro.train \
  --base-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --dataset /tmp/mios-micro-sft.jsonl \
  --output-dir ./output
```

### 4. Convert & Quantize to GGUF
```bash
python3 -m mios_micro.convert \
  --model-dir ./output \
  --quant-type Q4_K_M \
  --output mios-micro-1.5b-q4_k_m.gguf
```

### 5. Validate & Benchmark
```bash
python3 -m mios_micro.eval --model mios-micro-1.5b-q4_k_m.gguf
```

---

## 6. OpenAI API Wire Usage

### Structured Log Triage
```bash
curl -s http://localhost:8500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mios-micro:1.5b",
    "messages": [
      {
        "role": "system",
        "content": "You are MiOS-Micro Log Triager. Output structured JSON."
      },
      {
        "role": "user",
        "content": "bootc[412]: composefs digest mismatch on commit 8f192bc"
      }
    ],
    "response_format": {"type": "json_object"}
  }' | jq .
```

**Output:**
```json
{
  "severity": "WARN",
  "subsystem": "bootc",
  "root_cause": "composefs digest verification mismatch",
  "actionable": true,
  "recommended_action": "bootc rollback"
}
```

---

## 7. License

Distributed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.