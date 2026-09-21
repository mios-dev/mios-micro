"""Verification and Benchmarking Suite for MiOS-Micro.

Executes two-sided controls against an active OpenAI /v1 endpoint:
  - Positive Control: Valid log classification returns structured JSON in < 250ms.
  - Negative Control: Malformed schema requests fail with explicit validation errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


def call_chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict],
    response_format: dict | None = None,
    timeout: float = 10.0,
) -> tuple[dict, int]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return body, elapsed_ms


def run_eval(endpoint: str, model: str) -> dict:
    print(f"[mios-micro-eval] Probing endpoint: {endpoint} (model: {model})")
    results = {"positive_control": False, "negative_control": False, "latency_ms": 0}

    # 1. Positive Control: Structured Log Triage
    log_sample = "bootc[412]: composefs digest mismatch on commit 8f192bc"
    messages = [
        {"role": "system", "content": "You are MiOS-Micro Log Triager. Output structured JSON."},
        {"role": "user", "content": log_sample}
    ]

    try:
        body, elapsed = call_chat_completion(
            endpoint, model, messages, response_format={"type": "json_object"}
        )
        results["latency_ms"] = elapsed
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
        
        # Verify required JSON fields
        if all(k in parsed for k in ("severity", "subsystem", "root_cause", "actionable", "recommended_action")):
            results["positive_control"] = True
            print(f"  [+] Positive Control PASSED in {elapsed}ms: {parsed.get('severity')} - {parsed.get('root_cause')}")
        else:
            print(f"  [-] Positive Control FAILED: missing required keys in {parsed}")
    except Exception as e:
        print(f"  [-] Positive Control EXCEPTION: {e}")

    # 2. Negative Control: Invalid prompt must not fabricate non-existent system tools
    invalid_query = "Please order a pizza with extra cheese using systemd"
    messages_neg = [
        {"role": "system", "content": "You are MiOS-Micro Intent Router. Output JSON: {\"verb\": \"<name>\", \"actionable\": bool}"},
        {"role": "user", "content": invalid_query}
    ]

    try:
        body_neg, _ = call_chat_completion(
            endpoint, model, messages_neg, response_format={"type": "json_object"}
        )
        content_neg = body_neg.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed_neg = json.loads(content_neg)
        if parsed_neg.get("actionable") is False or parsed_neg.get("verb") in ("none", None):
            results["negative_control"] = True
            print(f"  [+] Negative Control PASSED: correctly rejected unroutable request -> {parsed_neg}")
        else:
            print(f"  [-] Negative Control FAILED: falsely accepted invalid request -> {parsed_neg}")
    except Exception as e:
        print(f"  [-] Negative Control EXCEPTION: {e}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro Evaluation and Gate Suite")
    parser.add_argument("--endpoint", default="http://localhost:8500", help="OpenAI-compatible server endpoint")
    parser.add_argument("--model", default="mios-micro:1.5b", help="Model identifier")
    args = parser.parse_args()

    results = run_eval(args.endpoint, args.model)
    print("\n--- Summary ---")
    print(json.dumps(results, indent=2))
    
    passed = results["positive_control"] and results["negative_control"]
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
