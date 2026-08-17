"""Tests for scripts/summarize_results.py, resolving P1 issue "eval: tach
baseline va proposed method trong benchmark" (method x ratio x task table
generation). Uses a synthetic results.json shaped like VCCBench.evaluate()'s
real output (see docs/benchmark.md's "Result JSON schema") rather than a
real model run, so this stays CPU-only and fast.
"""
import json

import pytest

from scripts.summarize_results import build_rows, format_csv, format_markdown, load_results

SAMPLE_RESULTS = {
    "none": {
        "long_document_qa": {
            "ratio_2.0": {"mean_compression_ratio": 1.0, "mean_quality_score": 1.0, "num_samples": 10},
        },
    },
    "random": {
        "long_document_qa": {
            "ratio_2.0": {"mean_compression_ratio": 2.0, "mean_quality_score": 0.4, "num_samples": 10},
        },
    },
    "combined": {
        "long_document_qa": {
            "ratio_2.0": {
                "mean_compression_ratio": 2.0, "mean_quality_score": 0.7,
                "mean_tone_preservation_rate": 0.9, "num_samples": 10,
            },
        },
    },
    "summary": {"note": "must be skipped, not a method"},
}


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    with open(d / "vcc_bench_results.json", "w", encoding="utf-8") as f:
        json.dump(SAMPLE_RESULTS, f)
    return str(d)


def test_load_results_reads_the_file(results_dir):
    results = load_results(results_dir)
    assert "none" in results
    assert "summary" in results


def test_load_results_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(str(tmp_path))


def test_build_rows_skips_summary_key(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    methods = {r["method"] for r in rows}
    assert "summary" not in methods
    assert methods == {"none", "random", "combined"}


def test_build_rows_tags_categories_correctly(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    by_method = {r["method"]: r["category"] for r in rows}
    assert by_method["none"] == "baseline"
    assert by_method["random"] == "baseline"
    assert by_method["combined"] == "proposed"


def test_build_rows_sorted_baseline_before_proposed(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    categories_in_order = [r["category"] for r in rows]
    assert categories_in_order.index("baseline") < categories_in_order.index("proposed")


def test_build_rows_extracts_metric_columns(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    combined_row = next(r for r in rows if r["method"] == "combined")
    assert combined_row["mean_tone_preservation_rate"] == 0.9
    assert combined_row["task"] == "long_document_qa"
    assert combined_row["ratio"] == "2.0"


def test_build_rows_missing_metric_is_none(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    none_row = next(r for r in rows if r["method"] == "none")
    # 'none' baseline never reports tone_preservation_rate.
    assert none_row["mean_tone_preservation_rate"] is None


def test_format_markdown_contains_category_separators(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    md = format_markdown(rows)
    assert "**baseline**" in md
    assert "**proposed**" in md
    assert "combined" in md


def test_format_markdown_empty_rows():
    assert "no results" in format_markdown([])


def test_format_csv_round_trips_row_count(results_dir):
    results = load_results(results_dir)
    rows = build_rows(results, context="registry")
    csv_text = format_csv(rows)
    lines = [line for line in csv_text.strip().splitlines() if line]
    assert len(lines) == len(rows) + 1  # +1 header
