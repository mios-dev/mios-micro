<!-- AI-hint: End-User License Agreement (EULA) for mios-micro — defining terms of use, deployment guidelines, liability boundaries, and acceptable use policies under Apache-2.0.
     AI-related: LICENSE, LEGAL.md, AGENTS.md -->
# End-User License Agreement (EULA)

**Last Updated: September 2026**

IMPORTANT: PLEASE READ THIS END-USER LICENSE AGREEMENT ("AGREEMENT" OR "EULA") CAREFULLY BEFORE DOWNLOADING, ACCESSING, DEPLOYING, OR USING **MIOS-MICRO** (INCLUDING ITS SOURCE CODE, TRAINED WEIGHTS, GGUF BINARIES, CONTAINER IMAGES, OR ASSOCIATED DATASETS).

BY DOWNLOADING, INSTALLING, RUNNING, OR USING MIOS-MICRO, YOU AGREE TO BE BOUND BY THE TERMS OF THIS AGREEMENT. IF YOU DO NOT AGREE TO THESE TERMS, DO NOT INSTALL, ACCESS, OR USE MIOS-MICRO.

---

## 1. Governance & Apache 2.0 Incorporation

1.1. **Open Source Core**: MiOS-Micro is free and open-source software (FOSS). This Agreement supplements and incorporates the terms and conditions of the **Apache License, Version 2.0** ("Apache 2.0"). The full text of the Apache 2.0 License is provided in the [LICENSE](LICENSE) file.

1.2. **Conflict of Terms**: In the event of any irreconcilable conflict between the terms of this EULA and the Apache 2.0 License regarding software and copyright permissions, the terms of the Apache 2.0 License shall prevail to the extent of such conflict.

---

## 2. Grant of Rights

Subject to the terms and conditions of this Agreement and the Apache 2.0 License, the MiOS contributors hereby grant you a perpetual, worldwide, non-exclusive, no-charge, royalty-free license to:
* Deploy, execute, and host the MiOS-Micro model locally, on edge devices, or within cloud container infrastructures.
* Modify, fine-tune, distill, quantize, and adapt the model weights and source code.
* Package and distribute the model as an OCI Container Image, OCI Artifact, or standalone GGUF binary.
* Integrate MiOS-Micro into commercial, proprietary, academic, or open-source software applications.

---

## 3. Autonomous Execution & Operator Responsibility

3.1. **Probabilistic Generation**: MiOS-Micro is a generative neural model trained to assist in shell command synthesis, system log triage, and tool dispatching. You acknowledge and agree that generative artificial intelligence produces probabilistic outputs that may occasionally be erroneous, outdated, or incomplete.

3.2. **Operator Oversight**: 
* YOU ACKNOWLEDGE THAT EXECUTING SYSTEM COMMANDS, ELEVATED SCRIPTS, CONTAINER LIFECYCLE ACTIONS, OR STORAGE OPERATIONS SYNTHESIZED BY MIOS-MICRO CARRIES INHERENT OPERATIONAL RISKS.
* The human operator or controlling application is solely responsible for implementing adequate execution gates, sanity checks, dry-run validations, and security boundaries before executing commands with system-level privileges (`sudo`, `wheel`, root).
* The authors and contributors of MiOS-Micro assume no responsibility or liability for unauthorized file deletions, service interruptions, data corruption, or system outages resulting from unverified command execution.

---

## 4. Acceptable Use Policy

You agree not to use MiOS-Micro, its derived weights, or associated container artifacts:
1. In violation of any applicable local, state, national, or international law, regulation, or treaty.
2. For the purpose of exploiting, harming, or attempting to exploit or harm minors in any way.
3. To generate, develop, or facilitate malicious software, ransomware, cyberweapons, rootkits, denial-of-service tools, or automated exploits against third-party systems without prior explicit authorization.
4. To conduct unauthorized biometric identification, unlawful surveillance, or deceptive impersonation.
5. To intentionally bypass, disable, or tamper with system safety controls, hardware isolation boundaries, or digital signatures required by the underlying host operating system.

---

## 5. Disclaimer of Warranties

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW:
* MIOS-MICRO IS PROVIDED ON AN **"AS IS" AND "AS AVAILABLE" BASIS**, WITH ALL FAULTS AND WITHOUT WARRANTY OF ANY KIND.
* THE CONTRIBUTORS, AUTHORS, AND COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, ACCURACY, AND NON-INFRINGEMENT.
* NO ORAL OR WRITTEN ADVICE, TELEMETRY, OR OUTPUT PROVIDED BY THE SOFTWARE OR MODEL SHALL CREATE A WARRANTY NOT EXPRESSLY STATED IN THIS AGREEMENT.

---

## 6. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW:
* IN NO EVENT SHALL ANY CONTRIBUTOR, COPYRIGHT HOLDER, OR AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; SYSTEM DOWNTIME; BUSINESS INTERRUPTION; OR HARDWARE FAILURE) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF, OR INABILITY TO USE, MIOS-MICRO, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## 7. Termination

This Agreement is effective until terminated. Your rights under this Agreement will terminate automatically without notice if you fail to comply with any of its material terms (including Section 4, Acceptable Use). Upon termination, you shall cease all use of the software, weights, and artifacts, and destroy all copies in your possession.

---

## 8. General Provisions & Severability

If any provision of this Agreement is held to be invalid, illegal, or unenforceable, the validity, legality, and enforceability of the remaining provisions shall not in any way be affected or impaired. This Agreement constitutes the complete understanding between the parties regarding the subject matter hereof.
