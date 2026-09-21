"""4-Pillar Dataset Generator for MiOS-Micro.

Synthesizes the training corpus across the four core MiOS operational pillars:
  1. System Command & CLI Synthesis (35%)
  2. Journal Log Event Triage (25%)
  3. Fast Intent & Capability Routing (25%)
  4. Strict Function Calling & Schema Conformance (15%)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Pillar System Prompts
SYS_PROMPT_CLI = (
    "You are MiOS-Micro CLI Assistant. Given an administrative or operational request on "
    "an immutable bootc/Fedora system, output ONLY the exact shell command to execute."
)

SYS_PROMPT_LOG = (
    "You are MiOS-Micro Log Triager. Given a journal or system log line, output ONLY a "
    "JSON object with keys: severity (INFO|WARN|ERROR|CRITICAL), subsystem, root_cause, "
    "actionable (true/false), and recommended_action."
)

SYS_PROMPT_ROUTER = (
    "You are MiOS-Micro Intent Router. Given a user request, classify it to the exact system "
    "verb name from the catalog or 'none'. Output ONLY a JSON object: {\"verb\": \"<name>\", \"actionable\": true/false}."
)

SYS_PROMPT_TOOLS = (
    "You are MiOS-Micro Function Calling Agent. Output standard OpenAI tool calls for valid requests."
)

# Baseline Seed Corpus (Deterministic foundation guaranteeing training integrity offline)
SEEDS_CLI = [
    ("Check system health and bootc deployment status", "bootc status"),
    ("Rollback to the previous deployment after bad update", "bootc rollback"),
    ("Upgrade host system to latest bootc container image", "bootc upgrade"),
    ("Inspect all failed systemd units on the host", "systemctl --failed"),
    ("Check journal errors from previous boot", "journalctl -b -1 -p err"),
    ("View live container logs for mios-llm-light", "podman logs -f mios-llm-light"),
    ("Verify Greenboot health checks manually", "greenboot check"),
    ("List all running podman containers", "podman ps"),
    ("Run OpenSCAP compliance scan on the node", "mios oscap"),
    ("Find all rust files in workspace", "find . -name '*.rs'"),
    ("Inspect composefs status on current deployment", "cat /sys/fs/composefs/status 2>/dev/null || bootc status"),
    ("Restart network manager service", "systemctl restart NetworkManager"),
    ("Check free disk space on root and persistent var", "df -h / /var"),
    ("Verify integrity of ostree commits", "ostree fsck"),
]

SEEDS_LOG = [
    ("systemd[1]: Failed to start mios-gateway-agent.service: Unit entered failed state.",
     {"severity": "ERROR", "subsystem": "systemd", "root_cause": "service failed to start", "actionable": True, "recommended_action": "systemctl status mios-gateway-agent.service"}),
    ("kernel: [1204.12] NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
     {"severity": "CRITICAL", "subsystem": "nvidia", "root_cause": "GPU fallen off bus", "actionable": True, "recommended_action": "systemctl restart nvidia-persistenced || reboot"}),
    ("bootc[412]: composefs digest mismatch on commit 8f192bc",
     {"severity": "WARN", "subsystem": "bootc", "root_cause": "composefs digest verification mismatch", "actionable": True, "recommended_action": "bootc rollback"}),
    ("greenboot[312]: Health check 01_network_check.sh passed successfully.",
     {"severity": "INFO", "subsystem": "greenboot", "root_cause": "health check passed", "actionable": False, "recommended_action": "none"}),
    ("podman[1092]: Error: container mios-pgvector exited with status 137 (OOMKilled)",
     {"severity": "ERROR", "subsystem": "podman", "root_cause": "container killed due to out of memory", "actionable": True, "recommended_action": "podman restart mios-pgvector"}),
    ("kernel: BTRFS info (device nvme0n1p3): balance: start -d -m",
     {"severity": "INFO", "subsystem": "storage", "root_cause": "filesystem maintenance active", "actionable": False, "recommended_action": "none"}),
    ("NetworkManager[850]: <warn> [1718921.1] dhcp4 (eth0): request timed out",
     {"severity": "WARN", "subsystem": "network", "root_cause": "DHCP lease acquisition failure", "actionable": True, "recommended_action": "systemctl restart NetworkManager"}),
]

SEEDS_ROUTER = [
    ("Find where the python config is located", {"verb": "find_file", "actionable": True}),
    ("Build the container image inside MiOS-DEV", {"verb": "build", "actionable": True}),
    ("Show me current system health and resources", {"verb": "status", "actionable": True}),
    ("What is the meaning of life?", {"verb": "none", "actionable": False}),
    ("Tell me a funny joke about Linux", {"verb": "none", "actionable": False}),
    ("Open the configurator dashboard in browser", {"verb": "dash", "actionable": True}),
    ("Audit standing gate checks and drift rules", {"verb": "audit", "actionable": True}),
]

SEEDS_TOOLS = [
    {
        "user": "What is the status of the mios-llm-light container?",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "systemctl_status",
                    "description": "Get systemd service status",
                    "parameters": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}},
                        "required": ["service"]
                    }
                }
            }
        ],
        "assistant_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "systemctl_status",
                    "arguments": "{\"service\": \"mios-llm-light\"}"
                }
            }
        ]
    }
]


def make_turn(system: str, user: str, assistant: str | dict) -> dict:
    content = assistant if isinstance(assistant, str) else json.dumps(assistant, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": content},
        ]
    }


def generate_micro_dataset(count: int = 100) -> list[dict]:
    """Assemble balanced dataset conforming to the 35/25/25/15 ratio."""
    samples = []
    
    n_cli = int(count * 0.35)
    n_log = int(count * 0.25)
    n_route = int(count * 0.25)
    n_tools = count - (n_cli + n_log + n_route)

    # 1. CLI Synthesis (35%)
    for i in range(n_cli):
        u, a = SEEDS_CLI[i % len(SEEDS_CLI)]
        samples.append(make_turn(SYS_PROMPT_CLI, u, a))

    # 2. Log Triage (25%)
    for i in range(n_log):
        u, a = SEEDS_LOG[i % len(SEEDS_LOG)]
        samples.append(make_turn(SYS_PROMPT_LOG, u, a))

    # 3. Router (25%)
    for i in range(n_route):
        u, a = SEEDS_ROUTER[i % len(SEEDS_ROUTER)]
        samples.append(make_turn(SYS_PROMPT_ROUTER, u, a))

    # 4. Tools / Function Calling (15%)
    for i in range(n_tools):
        t = SEEDS_TOOLS[i % len(SEEDS_TOOLS)]
        samples.append({
            "messages": [
                {"role": "system", "content": SYS_PROMPT_TOOLS},
                {"role": "user", "content": t["user"]},
                {"role": "assistant", "content": "", "tool_calls": t["assistant_calls"]}
            ]
        })

    random.shuffle(samples)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro 4-Pillar Dataset Generator")
    parser.add_argument("--count", type=int, default=200, help="Total number of examples")
    parser.add_argument("--out", type=str, default="mios-micro-sft.jsonl", help="Output JSONL path")
    parser.add_argument("--stats", action="store_true", help="Print dataset pillar distribution")
    args = parser.parse_args()

    samples = generate_micro_dataset(args.count)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[mios-micro-dataset] Generated {len(samples)} examples -> {out_path}")
    if args.stats:
        print("  - CLI Synthesis: ~35%")
        print("  - Log Triage:    ~25%")
        print("  - Intent Route:  ~25%")
        print("  - Tool Calling:  ~15%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
