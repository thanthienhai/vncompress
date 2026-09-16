"""Tests for scripts/analyze_gate1.py -- the Gate-1 evidence builder.

The statistics here decide whether wave 1's 2.3% survives as a finding or is
retired as a benchmark artifact, so they are tested against synthetic runs
whose ground truth is known by construction. A GPU sweep cannot be a unit test;
a simulated one can.
"""
import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "analyze_gate1",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "analyze_gate1.py"),
)
analyze_gate1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_gate1)


def _needle_row(arm, idx, group, recall, ratio=4.0, **extra):
    row = {
        "method": arm,
        "task": "needle_in_haystack",
        "sample_id": f"needle_v2_{idx:04d}_{group}",
        "needle_group": group,
        "insert_position": ("beginning", "middle", "end")[idx % 3],
        "needle_recall": recall,
        "token_f1": recall,
        "compression_ratio": ratio,
        "requested_ratio": ratio,
        "processing_time_ms": 10.0,
        "generation_time_ms": 20.0,
        "run": "gen1",
    }
    row.update(extra)
    return row


def _write_run(tmp_path, rows, run_name="gen1"):
    """Lay rows out the way benchmark.py writes them, so the loader is tested too."""
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for r in rows:
        grouped.setdefault((r["method"], r["task"], r["requested_ratio"]), []).append(r)
    for (arm, task, ratio), rs in grouped.items():
        (run_dir / f"{arm}_{task}_ratio{ratio:.1f}_seed42.json").write_text(
            json.dumps(rs), encoding="utf-8"
        )
    # Files benchmark.py also drops in the directory and the loader must skip.
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "vcc_bench_results.json").write_text("{}", encoding="utf-8")
    return run_dir


class TestLoadRuns:
    def test_skips_non_result_files_and_tags_the_run(self, tmp_path):
        run_dir = _write_run(tmp_path, [_needle_row("lacc_tone", i, "A", 0.1) for i in range(3)])
        rows = analyze_gate1.load_runs([str(run_dir)])
        assert len(rows) == 3
        assert {r["run"] for r in rows} == {run_dir.name}


def test_collapse_random_seeds():
    assert analyze_gate1.collapse_random_seeds("random_s1") == "random"
    assert analyze_gate1.collapse_random_seeds("random_s17") == "random"
    assert analyze_gate1.collapse_random_seeds("lacc_tone") == "lacc_tone"
    # Not a seeded arm -- must not be folded into `random`.
    assert analyze_gate1.collapse_random_seeds("sentence_retrieval") == "sentence_retrieval"


class TestTableC1:
    """The hypothesis of docs/eval_sweep_gate1.md SS5, simulated both ways."""

    @staticmethod
    def _render(rows):
        lines = []
        analyze_gate1.needle_group_table(rows, lines)
        return "\n".join(lines)

    def test_detects_a_tone_driven_collapse_on_group_a(self):
        # Ground truth: the arm fails on undiacriticized needles, holds on
        # diacriticized ones. The paired A-vs-B delta must come out large,
        # positive and significant.
        rows = []
        for i in range(40):
            rows.append(_needle_row("lacc_tone", i, "A", 0.02 + 0.001 * (i % 5)))
            rows.append(_needle_row("lacc_tone", i, "B", 0.85 + 0.001 * (i % 5)))
        out = self._render(rows)
        line = next(ln for ln in out.splitlines() if "`lacc_tone`" in ln)
        assert "+0.8" in line, line
        assert "*" in line, f"a 0.83 gap over 40 pairs must be significant: {line}"
        assert "| 40 |" in line

    def test_reports_no_group_effect_for_a_tone_blind_arm(self):
        # llmlingua has no tone signal: same performance on A and B, so the
        # delta must be ~0 and NOT flagged significant. If this ever fires,
        # the analysis manufactures the result it was written to look for.
        rows = []
        for i in range(40):
            rows.append(_needle_row("llmlingua", i, "A", 0.60 + 0.01 * (i % 7)))
            rows.append(_needle_row("llmlingua", i, "B", 0.60 + 0.01 * ((i + 3) % 7)))
        out = self._render(rows)
        line = next(ln for ln in out.splitlines() if "`llmlingua`" in ln)
        assert "*" not in line.split("|")[5], f"no real effect must not be starred: {line}"

    def test_pairs_by_haystack_not_by_row_order(self):
        # A and B rows are emitted in a different order on purpose: pairing must
        # follow the sample_id index, not the position in the file. Getting this
        # wrong silently compares unrelated haystacks.
        rows = [_needle_row("lacc_tone", i, "A", i / 40) for i in range(40)]
        rows += [_needle_row("lacc_tone", i, "B", i / 40) for i in reversed(range(40))]
        out = self._render(rows)
        line = next(ln for ln in out.splitlines() if "`lacc_tone`" in ln)
        # Correctly paired, every pair has delta exactly 0.
        assert "+0.000" in line, line

    def test_averages_random_seeds_before_pairing(self):
        rows = []
        for i in range(10):
            rows.append(_needle_row("random_s1", i, "A", 0.0))
            rows.append(_needle_row("random_s2", i, "A", 1.0))
            rows.append(_needle_row("random_s1", i, "B", 0.5))
            rows.append(_needle_row("random_s2", i, "B", 0.5))
        out = self._render(rows)
        line = next(ln for ln in out.splitlines() if "`random`" in ln)
        # Seeds averaged: A = 0.5, B = 0.5, delta 0 -- not three separate arms.
        assert "   0.500 " in line, line
        assert "+0.000" in line, line

    def test_handles_missing_group_without_crashing(self):
        rows = [_needle_row("lacc_tone", i, "A", 0.1) for i in range(5)]
        out = self._render(rows)
        assert "`lacc_tone`" in out


class TestRunHealth:
    def test_flags_failed_generations(self):
        rows = [_needle_row("lacc_tone", i, "A", None, generation_error="OutOfMemoryError: x")
                for i in range(3)]
        lines = []
        healthy = analyze_gate1.report_generation_health(rows, lines)
        assert healthy is False
        assert "3 / 3" in "\n".join(lines)

    def test_clean_run_reports_healthy(self):
        rows = [_needle_row("lacc_tone", i, "A", 0.5) for i in range(3)]
        lines = []
        assert analyze_gate1.report_generation_health(rows, lines) is True


class TestRealizedCompression:
    def test_flags_arms_whose_realized_ratio_misses_the_request(self):
        # Wave 1 saw LLMLingua realize 4.70x when asked for 4x. An arm compared
        # at its configured ratio is compared at the wrong operating point.
        rows = [_needle_row("llmlingua", i, "A", 0.5, ratio=4.0) for i in range(5)]
        for r in rows:
            r["compression_ratio"] = 7.3  # requested 4.0
        lines = []
        analyze_gate1.realized_cr_tables(rows, lines)
        out = "\n".join(lines)
        assert "misses the request by >15%" in out
        assert "1.82" in out  # 7.3 / 4.0

    def test_quiet_when_arms_hit_their_target(self):
        rows = [_needle_row("lacc_sentence", i, "A", 0.5, ratio=4.0) for i in range(5)]
        lines = []
        analyze_gate1.realized_cr_tables(rows, lines)
        assert "misses the request" not in "\n".join(lines)


class TestCrossGenerator:
    def test_detects_a_rank_inversion_between_generators(self):
        # A conclusion that flips between generators is a conclusion about the
        # generator (docs/eval_sweep_gate1.md SS4 rule 2).
        rows = []
        for i in range(5):
            rows.append(_needle_row("lacc_sentence", i, "A", 0.9, run="gen1"))
            rows.append(_needle_row("llmlingua", i, "A", 0.1, run="gen1"))
            rows.append(_needle_row("lacc_sentence", i, "A", 0.1, run="gen2"))
            rows.append(_needle_row("llmlingua", i, "A", 0.9, run="gen2"))
        lines = []
        analyze_gate1.cross_generator_check(rows, lines)
        out = "\n".join(lines)
        assert "Rank inversions" in out
        assert "`lacc_sentence` vs `llmlingua`" in out

    def test_stable_ordering_is_reported_as_stable(self):
        rows = []
        for i in range(5):
            for run, base in (("gen1", 0.9), ("gen2", 0.8)):
                rows.append(_needle_row("lacc_sentence", i, "A", base, run=run))
                rows.append(_needle_row("llmlingua", i, "A", base - 0.3, run=run))
        lines = []
        analyze_gate1.cross_generator_check(rows, lines)
        assert "No rank inversions" in "\n".join(lines)


def test_end_to_end_writes_a_report(tmp_path, monkeypatch):
    rows = []
    for i in range(40):
        rows.append(_needle_row("lacc_tone", i, "A", 0.02))
        rows.append(_needle_row("lacc_tone", i, "B", 0.85))
        rows.append(_needle_row("llmlingua", i, "A", 0.60))
        rows.append(_needle_row("llmlingua", i, "B", 0.61))
    run_dir = _write_run(tmp_path, rows)
    out_dir = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["analyze_gate1.py", "--runs", str(run_dir), "--out", str(out_dir)])
    assert analyze_gate1.main() == 0
    report = (out_dir / "gate1_analysis.md").read_text(encoding="utf-8")
    assert "Table C1" in report
    assert "`lacc_tone`" in report and "`llmlingua`" in report


def test_main_exits_nonzero_when_generations_failed(tmp_path, monkeypatch):
    # A broken sweep must not quietly produce a clean-looking report.
    rows = [_needle_row("lacc_tone", i, "A", None, generation_error="RuntimeError: CUDA OOM")
            for i in range(4)]
    run_dir = _write_run(tmp_path, rows)
    monkeypatch.setattr(sys, "argv",
                        ["analyze_gate1.py", "--runs", str(run_dir), "--out", str(tmp_path / "r")])
    assert analyze_gate1.main() == 1


def test_main_errors_on_empty_run_dir(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", ["analyze_gate1.py", "--runs", str(empty), "--out", str(tmp_path / "r")])
    with pytest.raises(SystemExit):
        analyze_gate1.main()
