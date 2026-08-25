import tempfile
import unittest
from pathlib import Path

from ergix_hormozi.store import KnowledgeStore, chunk_text


class StoreTests(unittest.TestCase):
    def test_chunk_text_preserves_all_sections(self):
        text = "First paragraph.\n\n" + ("Second paragraph with useful detail. " * 30)
        chunks = chunk_text(text, size=220, overlap=30)
        self.assertGreater(len(chunks), 2)
        self.assertIn("First paragraph", chunks[0])
        self.assertTrue(any("useful detail" in chunk for chunk in chunks))

    def test_ingest_deduplicates_and_searches(self):
        with tempfile.TemporaryDirectory() as directory:
            store = KnowledgeStore(Path(directory) / "knowledge.db")
            first = store.ingest("memo.md", "Offer memo", "A strong offer reduces perceived time delay and increases certainty.")
            second = store.ingest("memo.md", "Offer memo", "A strong offer reduces perceived time delay and increases certainty.")
            hits = store.search("perceived time delay", top_k=3)
            self.assertEqual(first["status"], "indexed")
            self.assertEqual(second["status"], "unchanged")
            self.assertEqual(hits[0].title, "Offer memo")
            store.close()


if __name__ == "__main__":
    unittest.main()
