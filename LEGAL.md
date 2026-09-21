<!-- AI-hint: Legal notices, intellectual property attribution, FOSS governance, and compliance documentation for mios-micro.
     AI-related: LICENSE, EULA.md, AGENTS.md, /usr/share/mios/mios.toml -->
# Legal Notices & Intellectual Property Attribution

This document outlines the legal notices, open-source licensing compliance, copyright ownership, trademark disclaimers, and upstream intellectual property attributions for the **MiOS-Micro** repository, its fine-tuning pipeline, datasets, container images, and binary model artifacts (GGUF).

---

## 1. Primary License

The **MiOS-Micro** source code, fine-tuning scripts, evaluation harnesses, Containerfiles, and orchestration tools are licensed under the **Apache License, Version 2.0** (the "License"). You may obtain a copy of the License in the [LICENSE](LICENSE) file or at:

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software, models, and datasets distributed under the License are distributed on an **"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See the License for the specific language governing permissions and limitations.

---

## 2. Upstream Base Model Attribution

MiOS-Micro is built via supervised fine-tuning (LoRA SFT) and distillation upon the following foundational model:

* **Model Name**: `Qwen2.5-Coder-1.5B-Instruct`
* **Original Author / Copyright**: Alibaba Cloud (<https://github.com/QwenLM/Qwen2.5-Coder>)
* **Upstream License**: **Apache License 2.0**
* **Strict FOSS Compliance**:
  Unlike certain derivative variants governed by restrictive research-only licenses, the `1.5B` parameter base (`Qwen2.5-Coder-1.5B-Instruct`) was explicitly selected for its full OSI-compliant **Apache-2.0** license, ensuring complete freedom for commercial, educational, academic, and private self-hosting without proprietary restrictions.

---

## 3. Dataset Licensing & Provenance

* **Synthetic Self-Distillation**:
  All training datasets generated via `src/mios_micro/dataset.py` (including the 4-pillar corpus covering system command synthesis, journal log triage, intent routing, and OpenAI tool calling) are created via programmatic extraction of open-source system catalogs, standard POSIX/FHS commands, and synthetic self-distillation.
* **License of Datasets**:
  All resulting `.jsonl` dataset artifacts authored within this repository are dedicated to the public domain or released under the **Apache-2.0** license.
* **Zero PII & Data Privacy**:
  No Personally Identifiable Information (PII), proprietary private credentials, API keys, or private communications are collected, scraped, or embedded in the dataset generation pipelines.

---

## 4. Trademarks & Brand Usage

* **MiOS™**:
  "MiOS", "MiOS-Micro", and the MiOS project logos are trademarks or project identifiers of the MiOS Open Source Project. Use of these marks for descriptive purposes, attribution, and compatibility statements is permitted. Modifying or redistributing altered versions of the model or operating system in a way that misrepresents endorsement by the MiOS project is prohibited.
* **Third-Party Trademarks**:
  * "Linux" is a registered trademark of Linus Torvalds.
  * "Fedora" and "Red Hat" are trademarks or registered trademarks of Red Hat, Inc.
  * "Alibaba", "Alibaba Cloud", and "Qwen" are trademarks of Alibaba Group Holding Limited.
  * "Python" is a trademark of the Python Software Foundation.
  * "Docker", "Podman", "GitHub", and all other company, brand, or product names mentioned herein are property of their respective owners. Their mention does not imply sponsorship, affiliation, or endorsement.

---

## 5. Export Regulations & Jurisdiction

The source code, build scripts, and model weights provided in this repository are developed as freely available open-source software and published publicly without national discrimination, conforming to U.S. Export Administration Regulations (EAR) provisions for publicly available technology and software (15 CFR § 734.7) and comparable international export control frameworks. Operators and users remain individually responsible for complying with the local laws of their respective jurisdictions.

---

## 6. Responsible AI & Dual-Use Notice

MiOS-Micro is an edge-optimized miniature neural network designed for system automation, event triage, shell command synthesis, and fast capability routing. As with any machine learning system:
1. **Probabilistic Outputs**: Model outputs are inherently probabilistic and may occasionally contain inaccuracies, hallucinations, or syntactic errors.
2. **Execution Safeguards**: In accordance with MiOS Architectural Laws, automated execution of system commands or state-altering actions must pass through appropriate verification gates, user confirmation prompts, or sandbox protections.
