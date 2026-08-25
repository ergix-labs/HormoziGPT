import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ergix_hormozi.config import _read_harness_secret


class ConfigTests(unittest.TestCase):
    def test_reads_synced_secret_from_configured_harness_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "harness.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE secrets (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO secrets (name, value) VALUES (?, ?)",
                ("MOONSHOT_API_KEY", "synced-test-key"),
            )
            connection.commit()
            connection.close()

            with patch.dict(os.environ, {"HORMOZI_HARNESS_DB": str(database)}):
                self.assertEqual(_read_harness_secret("MOONSHOT_API_KEY"), "synced-test-key")


if __name__ == "__main__":
    unittest.main()
