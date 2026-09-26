"""Unit Tests and Two-Sided Verification Gates for mios_micro.eval."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from src.mios_micro import eval as micro_eval


class TestMicroEval(unittest.TestCase):

    def test_positive_control_valid_triage(self):
        """Positive Control: Valid structured JSON triage passes schema gate."""
        mock_response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "severity": "WARN",
                            "subsystem": "bootc",
                            "root_cause": "composefs digest verification mismatch",
                            "actionable": True,
                            "recommended_action": "bootc rollback"
                        })
                    }
                }
            ]
        }

        with patch("src.mios_micro.eval.call_chat_completion") as mock_call:
            # First call: positive triage, Second call: negative unroutable probe
            mock_call.side_effect = [
                (mock_response_body, 120),  # 120ms latency (< 250ms target)
                ({"choices": [{"message": {"content": json.dumps({"verb": "none", "actionable": False})}}]}, 80),
            ]
            results = micro_eval.run_eval("http://localhost:8500", "mios-micro:1.5b")
            self.assertTrue(results["positive_control"])
            self.assertTrue(results["negative_control"])
            self.assertLess(results["latency_ms"], 250)

    def test_negative_control_corrupt_schema_fails(self):
        """Negative Control: Missing required schema keys MUST fail the gate."""
        corrupt_response_body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "bad_key": "corrupted",
                            # missing severity, subsystem, root_cause, etc.
                        })
                    }
                }
            ]
        }

        with patch("src.mios_micro.eval.call_chat_completion") as mock_call:
            mock_call.side_effect = [
                (corrupt_response_body, 90),
                ({"choices": [{"message": {"content": json.dumps({"verb": "none", "actionable": False})}}]}, 80),
            ]
            results = micro_eval.run_eval("http://localhost:8500", "mios-micro:1.5b")
            self.assertFalse(results["positive_control"])

    def test_negative_control_hallucinated_tool_fails(self):
        """Negative Control: Hallucinating non-existent tools on unroutable prompts MUST fail."""
        with patch("src.mios_micro.eval.call_chat_completion") as mock_call:
            mock_call.side_effect = [
                ({"choices": [{"message": {"content": json.dumps({
                    "severity": "INFO", "subsystem": "test", "root_cause": "ok", "actionable": False, "recommended_action": "none"
                })}}]}, 50),
                ({"choices": [{"message": {"content": json.dumps({
                    "verb": "order_pizza_fake_tool", "actionable": True
                })}}]}, 60),
            ]
            results = micro_eval.run_eval("http://localhost:8500", "mios-micro:1.5b")
            self.assertFalse(results["negative_control"])


if __name__ == "__main__":
    unittest.main()
