import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from ergix_hormozi.daily import generate_daily_motions
from ergix_hormozi.store import KnowledgeStore


class FakeClient:
    chat_model = "test-model"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def chat(self, messages, json_mode=False):
        return json.dumps({"motions": [
            {"title": "Call five qualified leads", "why_now": "Pipeline is the constraint.", "action": "Call the five warmest leads.", "metric": "completed calls", "target": "5", "timebox_minutes": 45, "source_titles": ["Lead memo"]},
            {"title": "Tighten the offer", "why_now": "Clarity improves response.", "action": "Rewrite the offer in one sentence.", "metric": "offer variants", "target": "3", "timebox_minutes": 30, "source_titles": ["Offer memo"]},
            {"title": "Measure conversion", "why_now": "The baseline is missing.", "action": "Calculate the last 30-day close rate.", "metric": "close rate", "target": "1 measured baseline", "timebox_minutes": 20, "source_titles": []},
        ]})


class DailyTests(unittest.TestCase):
    def test_writes_daily_json_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = KnowledgeStore(root / "knowledge.db")
            store.ingest("lead.md", "Lead memo", "Call warm leads every day.", FakeClient().embed, "test")
            result = generate_daily_motions(store, FakeClient(), "Offer: consulting", root / "motions", day=date(2026, 8, 25))
            self.assertEqual(len(result["motions"]), 3)
            self.assertEqual(result["motions"][0]["target"], "5")
            self.assertEqual(result["motions"][0]["source_titles"], ["Lead memo"])
            self.assertEqual(result["motions"][1]["source_titles"], [])
            self.assertTrue((root / "motions" / "2026-08-25.json").exists())
            store.close()


if __name__ == "__main__":
    unittest.main()
