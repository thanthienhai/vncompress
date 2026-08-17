"""Guards P1 issue "data: document dataset provenance và versioning":
catches a dataset file changing without its checksum manifest being
regenerated, and checks the derived VCC-Bench files carry the provenance
metadata documented in vcc_bench_data/PROVENANCE.md.
"""
import json
import os

import pytest

from scripts.checksum_datasets import DATA_DIR, MANIFEST_PATH, compute_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_data_dir_resolves_to_vcc_bench_data():
    assert os.path.basename(DATA_DIR) == "vcc_bench_data"
    assert os.path.isdir(DATA_DIR)


def test_provenance_doc_exists():
    assert os.path.exists(os.path.join(DATA_DIR, "PROVENANCE.md"))


@pytest.mark.skipif(not os.path.exists(MANIFEST_PATH), reason="CHECKSUMS.json not generated")
def test_checksums_manifest_matches_files_on_disk():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    actual = compute_manifest()

    assert set(recorded) == set(actual), (
        "vcc_bench_data/CHECKSUMS.json is out of sync with the files on disk -- "
        "run `python scripts/checksum_datasets.py --write` and commit the update."
    )
    for name, expected in recorded.items():
        assert expected["sha256"] == actual[name]["sha256"], (
            f"{name} content changed but CHECKSUMS.json was not regenerated -- "
            "run `python scripts/checksum_datasets.py --write`."
        )


@pytest.mark.parametrize("filename", [
    "vcc_bench_v1.json",
    "vcc_bench_agent_tool_calling.json",
    "vcc_bench_cross_lingual.json",
    "vcc_bench_long_document_qa.json",
    "vcc_bench_multi_turn_conversation.json",
    "vcc_bench_needle_in_haystack.json",
])
def test_derived_dataset_has_provenance_metadata(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    for field in ("version", "date", "license"):
        assert field in meta, f"{filename} metadata missing required provenance field: {field}"


def test_raw_wikipedia_dataset_has_provenance_metadata():
    path = os.path.join(DATA_DIR, "wikipedia_vi_raw.json")
    if not os.path.exists(path):
        pytest.skip("wikipedia_vi_raw.json not present")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    for field in ("version", "date", "source", "license"):
        assert field in meta, f"wikipedia_vi_raw.json metadata missing required provenance field: {field}"
