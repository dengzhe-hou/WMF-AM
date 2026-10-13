#!/usr/bin/env python3
"""Offline regression checks using the observed Sonnet alias response."""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frozen"))
from check_api_records import endpoint_checks, read_jsonl, response_checks


class ApiIdentityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = "anthropic/claude-sonnet-4.5"
        cls.canonical = "anthropic/claude-4.5-sonnet-20250929"
        cls.provider = "Anthropic"
        cls.endpoint = json.loads((ROOT / "results/api/provenance/claude-sonnet-4-5-20250929.endpoints.20261003T151915Z.json").read_text())
        cls.records = read_jsonl(ROOT / "results/api/smoke/claude-sonnet-4-5-20250929.20261003T151915Z.jsonl")
        cls.ids = [item["item_id"] for item in read_jsonl(ROOT / "frozen/bundle/smoke_bank.jsonl")]

    def responses_pass(self, records):
        return all(response_checks(records, self.ids, self.request,
                                   self.canonical, self.provider, "smoke").values())

    def test_existing_alias_and_expected_canonical_pass(self):
        self.assertTrue(all(endpoint_checks(self.endpoint, self.request,
                                            self.canonical, self.provider).values()))
        self.assertTrue(self.responses_pass(self.records))
        records = copy.deepcopy(self.records)
        for record in records:
            record["response_model"] = self.canonical
        self.assertTrue(self.responses_pass(records))

    def test_wrong_response_identity_fails(self):
        for field, value in (("response_model", "anthropic/claude-haiku-4.5"),
                             ("request_model", "anthropic/claude-haiku-4.5"),
                             ("provider", "Amazon Bedrock")):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                self.assertFalse(self.responses_pass(records))

    def test_wrong_endpoint_identity_fails(self):
        for field in ("id", "canonical", "provider"):
            with self.subTest(field=field):
                endpoint = copy.deepcopy(self.endpoint)
                if field == "id":
                    endpoint["data"]["id"] = "anthropic/claude-haiku-4.5"
                else:
                    for entry in endpoint["data"]["endpoints"]:
                        if entry["provider_name"] == self.provider:
                            if field == "canonical":
                                entry["name"] = "Anthropic | anthropic/claude-4.5-sonnet-20990101"
                            else:
                                entry["provider_name"] = "Another provider"
                self.assertFalse(all(endpoint_checks(endpoint, self.request,
                                                     self.canonical, self.provider).values()))

    def test_reasoning_fails(self):
        for field in ("tokens", "trace"):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                if field == "tokens":
                    records[0]["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1}
                else:
                    records[0]["thinking_detected"] = True
                self.assertFalse(self.responses_pass(records))


if __name__ == "__main__":
    unittest.main()
