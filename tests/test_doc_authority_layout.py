"""Regression guards for the R12 documentation authority layout."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "authority" / "AUTHORITY_INDEX.json"
FROZEN_INDEX = DOCS / "experiments" / "FROZEN_INDEX.json"


class DocumentationAuthorityLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = json.loads(INDEX.read_text())

    def test_only_readme_lives_at_docs_root(self) -> None:
        root_files = sorted(p.name for p in DOCS.iterdir() if p.is_file())
        self.assertEqual(root_files, ["README.md"])

    def test_current_canonical_paths_are_isolated_and_exist(self) -> None:
        canonical = self.index["current_canonical"]
        self.assertTrue(canonical)
        for relative in canonical:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(path.parent, DOCS / "authority")

    def test_authority_classes_have_distinct_directories(self) -> None:
        directories = []
        for metadata in self.index["authority_classes"].values():
            path = ROOT / metadata["directory"]
            self.assertTrue(path.is_dir(), metadata["directory"])
            directories.append(path.resolve())
        self.assertEqual(len(directories), len(set(directories)))

    def test_removed_one_time_instructions_are_absent(self) -> None:
        for relative in self.index["removed_from_current_tree"]:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_frozen_and_archive_are_not_default_context(self) -> None:
        classes = self.index["authority_classes"]
        self.assertFalse(classes["FROZEN_HISTORICAL_AUTHORITY"]["default_context"])
        self.assertFalse(classes["SUPERSEDED_ARCHAEOLOGY"]["default_context"])
        self.assertTrue(classes["CURRENT_CANONICAL"]["default_context"])

    def test_historical_and_active_task_areas_are_separate(self) -> None:
        frozen = DOCS / "experiments" / "frozen"
        active = DOCS / "tasks" / "active"
        self.assertTrue((frozen / "R12_VS_D_HISTORICAL_MARKET_CANARY_R0.md").is_file())
        active_files = sorted(p.name for p in active.iterdir() if p.is_file())
        self.assertEqual(active_files, ["README.md"])

    def test_frozen_contract_hashes_are_immutable(self) -> None:
        frozen_index = json.loads(FROZEN_INDEX.read_text())
        self.assertEqual(frozen_index["schema"], "cb16.frozen-doc-index.v1")
        self.assertTrue(frozen_index["entries"])
        for entry in frozen_index["entries"]:
            path = ROOT / entry["current_path"]
            self.assertTrue(path.is_file(), entry["current_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["current_path"])


if __name__ == "__main__":
    unittest.main()
