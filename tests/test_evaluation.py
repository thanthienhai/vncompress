"""Tests for vncompress/evaluation.py (VCCBench, metrics, significance
testing, method taxonomy), vncompress/config.py (ExperimentConfig,
reproducibility), and evaluate.py's results-table logic. All CPU-only,
synthetic data -- no real model run or dataset download needed.
"""
import json
import os
import random

import pytest

from vncompress.config import (
    ExperimentConfig,
    load_experiment_config,
    save_run_metadata,
    set_seed,
    snapshot_environment,
)
from vncompress.evaluation import (
    ABLATION_ARM_CATEGORY,
    REGISTRY_METHOD_CATEGORY,
    CompressionMetrics,
    MethodCategory,
    VCCBench,
    VietnameseRougeTokenizer,
    categorize,
    compute_bleu,
    compute_exact_match,
    compute_needle_recall,
    compute_rouge_l,
    compute_token_f1,
    paired_bootstrap_delta,
)


# ============================================================================
# ExperimentConfig / reproducibility (vncompress/config.py)
# ============================================================================


def test_default_config_has_expected_fields():
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.device == "cuda"
    assert config.compression_ratios == [2.0, 4.0, 8.0]
    assert config.split == "full"


def test_resolved_tokenizer_defaults_to_model():
    assert ExperimentConfig(model="foo/bar").resolved_tokenizer() == "foo/bar"


def test_resolved_tokenizer_explicit_override():
    assert ExperimentConfig(model="foo/bar", tokenizer="baz/qux").resolved_tokenizer() == "baz/qux"


def test_load_experiment_config_with_no_args_returns_defaults():
    assert load_experiment_config() == ExperimentConfig()


def test_load_experiment_config_from_json_file(tmp_path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"seed": 7, "device": "cpu"}), encoding="utf-8")
    config = load_experiment_config(str(path))
    assert config.seed == 7
    assert config.device == "cpu"
    assert config.model == ExperimentConfig().model


def test_cli_overrides_take_precedence_over_config_file(tmp_path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"seed": 7, "device": "cpu"}), encoding="utf-8")
    config = load_experiment_config(str(path), cli_overrides={"device": "cuda"})
    assert config.seed == 7
    assert config.device == "cuda"


def test_cli_overrides_with_none_values_do_not_override():
    config = load_experiment_config(cli_overrides={"seed": None, "device": "cpu"})
    assert config.seed == ExperimentConfig().seed
    assert config.device == "cpu"


def test_unknown_field_in_config_file_raises(tmp_path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"not_a_real_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(str(path))


def test_unknown_field_in_cli_overrides_raises():
    with pytest.raises(ValueError):
        load_experiment_config(cli_overrides={"not_a_real_field": 1})


def test_set_seed_makes_random_deterministic():
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_snapshot_environment_has_expected_keys():
    snap = snapshot_environment()
    assert "timestamp_utc" in snap
    assert "git_commit" in snap
    assert "python_version" in snap
    assert "torch" in snap["package_versions"]


def test_save_run_metadata_writes_both_files(tmp_path):
    config = ExperimentConfig(seed=99, output_dir=str(tmp_path))
    save_run_metadata(str(tmp_path), config)
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved_config["seed"] == 99
    saved_env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert "git_commit" in saved_env


class TestExperimentConfigValidation:
    """Guards against settings that used to crash mid-run or produce
    plausible-looking nonsense (e.g. ratio=0 -> ZeroDivisionError only
    after the generation model had already loaded)."""

    @pytest.mark.parametrize("ratios", [[0.0], [-2.0], [0.5], [2.0, 0.0], []])
    def test_invalid_compression_ratios_are_rejected(self, ratios):
        with pytest.raises(ValueError, match="compression_ratios"):
            ExperimentConfig(compression_ratios=ratios)

    def test_ratio_one_is_allowed_as_a_no_compression_control(self):
        assert ExperimentConfig(compression_ratios=[1.0]).compression_ratios == [1.0]

    def test_empty_methods_is_rejected(self):
        with pytest.raises(ValueError, match="methods is empty"):
            ExperimentConfig(methods=[])

    def test_unknown_device_is_rejected(self):
        with pytest.raises(ValueError, match="device"):
            ExperimentConfig(device="banana")

    def test_unknown_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            ExperimentConfig(dtype="float8")


# ============================================================================
# Text metrics
# ============================================================================


class TestTokenF1:
    def test_identical_answers_score_one(self):
        assert compute_token_f1(["Phạm Văn Đồng"], ["Phạm Văn Đồng"]) == pytest.approx(1.0)

    def test_disjoint_answers_score_zero(self):
        assert compute_token_f1(["Hà Nội"], ["Sài Gòn"]) == pytest.approx(0.0)

    def test_partial_overlap_is_credited_where_exact_match_is_not(self):
        pred, ref = ["Thủ tướng Phạm Văn Đồng"], ["Phạm Văn Đồng"]
        assert compute_token_f1(pred, ref) == pytest.approx(0.75)
        assert compute_exact_match(pred, ref) == pytest.approx(0.0)

    def test_case_and_punctuation_are_normalized_away(self):
        assert compute_token_f1(["hà nội."], ["Hà Nội"]) == pytest.approx(1.0)

    def test_both_empty_agree_but_one_empty_does_not(self):
        assert compute_token_f1([""], [""]) == pytest.approx(1.0)
        assert compute_token_f1([""], ["Hà Nội"]) == pytest.approx(0.0)

    def test_diacritics_are_not_stripped(self):
        assert compute_token_f1(["ma"], ["mã"]) == pytest.approx(0.0)


class TestVietnameseRougeTokenization:
    """Regression guard for the tone-blind ROUGE bug: rouge_score's default
    tokenizer strips every character outside [a-z0-9], deleting Vietnamese
    tone marks, so 'bạn' scored 1.0 against 'bàn'."""

    def test_tone_marks_survive_tokenization(self):
        tok = VietnameseRougeTokenizer()
        assert tok.tokenize("bàn") == ["bàn"]
        assert tok.tokenize("bạn") == ["bạn"]
        assert tok.tokenize("Hà Nội là thủ đô") == ["hà", "nội", "là", "thủ", "đô"]

    def test_different_tones_are_not_scored_as_identical(self):
        score = compute_rouge_l(["bàn của tôi"], ["bạn của tôi"])["rougeL_f1"]
        assert score < 1.0

    def test_identical_vietnamese_text_still_scores_one(self):
        text = "Hà Nội là thủ đô của Việt Nam"
        assert compute_rouge_l([text], [text])["rougeL_f1"] == pytest.approx(1.0)

    def test_rouge_and_token_f1_count_the_same_units(self):
        pred, ref = ["Hà Nội là thủ đô"], ["Sài Gòn là thành phố"]
        assert compute_rouge_l(pred, ref)["rougeL_f1"] == pytest.approx(compute_token_f1(pred, ref))


class TestNeedleRecall:
    def test_recovered_needle_scores_one(self):
        assert compute_needle_recall(["mật khẩu bí mật là VIETCOMPRESS2026"], ["VIETCOMPRESS2026"]) == pytest.approx(1.0)

    def test_missing_needle_scores_zero(self):
        assert compute_needle_recall(["tôi không biết"], ["VIETCOMPRESS2026"]) == pytest.approx(0.0)

    def test_partial_needle_is_credited(self):
        assert compute_needle_recall(["Hà xxxx"], ["Hà Nội"]) == pytest.approx(0.5)

    def test_recall_ignores_surrounding_filler(self):
        assert compute_needle_recall(["câu trả lời đầy đủ là Hà Nội thủ đô"], ["Hà Nội"]) == pytest.approx(1.0)

    def test_tone_marks_are_not_collapsed(self):
        assert compute_needle_recall(["ma"], ["mã"]) == pytest.approx(0.0)

    def test_empty_predictions_does_not_crash(self):
        assert compute_needle_recall([], []) == pytest.approx(0.0)


def test_compute_bleu_identical_text_scores_high():
    text = "Hà Nội là thủ đô của Việt Nam"
    assert compute_bleu([text], [text]) > 0.9


# ============================================================================
# Paired bootstrap significance
# ============================================================================


class TestPairedBootstrap:
    def test_clear_improvement_is_significant(self):
        probe = [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.88, 0.72]
        rule = [0.5, 0.6, 0.4, 0.55, 0.65, 0.3, 0.58, 0.42]
        res = paired_bootstrap_delta(probe, rule, n_boot=2000, seed=0)
        assert res.mean_delta > 0
        assert res.ci_low > 0
        assert res.significant is True
        assert res.win_rate == pytest.approx(1.0)
        assert res.p_value < 0.05

    def test_no_difference_is_not_significant(self):
        xs = [0.5, 0.6, 0.4, 0.55, 0.65, 0.3]
        res = paired_bootstrap_delta(list(xs), list(xs), n_boot=2000, seed=0)
        assert res.mean_delta == pytest.approx(0.0)
        assert res.significant is False
        assert res.p_value == pytest.approx(1.0)

    def test_pairs_with_none_are_dropped(self):
        probe = [0.8, None, 0.7, 0.9]
        rule = [0.5, 0.6, None, 0.6]
        res = paired_bootstrap_delta(probe, rule, n_boot=1000, seed=0)
        assert res.n == 2

    def test_too_few_pairs_returns_none(self):
        assert paired_bootstrap_delta([0.5], [0.4]) is None
        assert paired_bootstrap_delta([None], [0.4]) is None

    def test_is_paired_not_unpaired(self):
        probe = [0.9, 0.1, 0.9, 0.1]
        rule = [0.9, 0.1, 0.9, 0.1]
        res = paired_bootstrap_delta(probe, rule, n_boot=1000, seed=0)
        assert res.mean_delta == pytest.approx(0.0)
        assert res.significant is False


# ============================================================================
# VCCBench aggregation regressions
# ============================================================================


def test_aggregate_never_emits_nan():
    # Was: np.mean([]) -> NaN, and json.dump writes a bare `NaN` token,
    # which is not valid JSON (RFC 8259).
    failed = [CompressionMetrics(compression_ratio=2.0) for _ in range(3)]
    agg = VCCBench()._aggregate_metrics(failed)
    assert agg["mean_rouge_l_f1"] is None
    assert agg["mean_bleu"] is None
    json.dumps(agg, allow_nan=False)


def test_harmonized_score_survives_negative_efficiency():
    # Was: 2*q*e/(q+e) with e<0 returned -18,000,000, sorting to the top of
    # a table labelled "higher is better".
    results = {"m": {"t": {"ratio_2.0": {
        "mean_quality_score": 0.3, "mean_efficiency_score": -0.3, "num_samples": 10,
    }}}}
    assert VCCBench()._compute_summary(results)["m"]["harmonized_score"] >= 0.0


def test_summary_weights_tasks_by_sample_count():
    # Was: plain mean over cells, so an 8-sample task counted as much as a
    # 160-sample one in the ranking table.
    results = {"m": {
        "big": {"ratio_2.0": {"mean_quality_score": 0.9, "mean_efficiency_score": 0.5, "num_samples": 1000}},
        "small": {"ratio_2.0": {"mean_quality_score": 0.1, "mean_efficiency_score": 0.5, "num_samples": 2}},
    }}
    avg_q = VCCBench()._compute_summary(results)["m"]["avg_quality"]
    assert avg_q == pytest.approx((0.9 * 1000 + 0.1 * 2) / 1002)
    assert avg_q > 0.85  # not the unweighted 0.5


# ============================================================================
# Method taxonomy (baseline / proposed / ablation)
# ============================================================================


def test_every_registry_method_is_classified():
    from vncompress.compression import METHODS

    missing = set(METHODS) - set(REGISTRY_METHOD_CATEGORY)
    assert not missing, f"METHODS missing from taxonomy: {missing}"


@pytest.mark.parametrize("name", ["none", "random", "llmlingua", "snapkv", "selective"])
def test_prior_art_methods_are_baseline(name):
    assert categorize(name, context="registry") == MethodCategory.BASELINE


def test_lacc_is_proposed_in_registry_context():
    assert categorize("lacc", context="registry") == MethodCategory.PROPOSED


@pytest.mark.parametrize("name", ["ppl_only", "tone_only", "morph_only", "lacc"])
def test_ablation_arms_are_ablation_in_ablation_context(name):
    assert categorize(name, context="ablation") == MethodCategory.ABLATION


def test_lacc_is_proposed_in_registry_but_ablation_in_ablation_context():
    # Same method name, different table -> different category, by design.
    assert categorize("lacc", context="registry") == MethodCategory.PROPOSED
    assert categorize("lacc", context="ablation") == MethodCategory.ABLATION


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        categorize("not_a_real_method", context="registry")


def test_invalid_context_raises():
    with pytest.raises(ValueError):
        categorize("none", context="not_a_real_context")


def test_ablation_arm_category_only_contains_ablation():
    assert set(ABLATION_ARM_CATEGORY.values()) == {MethodCategory.ABLATION}


# ============================================================================
# evaluate.py's results-table logic
# ============================================================================

SAMPLE_RESULTS = {
    "none": {
        "long_document_qa": {"ratio_2.0": {"mean_compression_ratio": 1.0, "mean_quality_score": 1.0, "num_samples": 10}},
    },
    "random": {
        "long_document_qa": {"ratio_2.0": {"mean_compression_ratio": 2.0, "mean_quality_score": 0.4, "num_samples": 10}},
    },
    "lacc": {
        "long_document_qa": {"ratio_2.0": {
            "mean_compression_ratio": 2.0, "mean_quality_score": 0.7,
            "mean_tone_preservation_rate": 0.9, "num_samples": 10,
        }},
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


def _evaluate_module():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluate_module():
    return _evaluate_module()


def test_load_results_reads_the_file(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    assert "none" in results
    assert "summary" in results


def test_load_results_missing_file_raises(tmp_path, evaluate_module):
    with pytest.raises(FileNotFoundError):
        evaluate_module.load_results(str(tmp_path))


def test_build_rows_skips_summary_key(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    methods = {r["method"] for r in rows}
    assert "summary" not in methods
    assert methods == {"none", "random", "lacc"}


def test_build_rows_tags_categories_correctly(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    by_method = {r["method"]: r["category"] for r in rows}
    assert by_method["none"] == "baseline"
    assert by_method["random"] == "baseline"
    assert by_method["lacc"] == "proposed"


def test_build_rows_sorted_baseline_before_proposed(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    categories_in_order = [r["category"] for r in rows]
    assert categories_in_order.index("baseline") < categories_in_order.index("proposed")


def test_build_rows_extracts_metric_columns(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    lacc_row = next(r for r in rows if r["method"] == "lacc")
    assert lacc_row["mean_tone_preservation_rate"] == 0.9
    assert lacc_row["task"] == "long_document_qa"
    assert lacc_row["ratio"] == "2.0"


def test_build_rows_missing_metric_is_none(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    none_row = next(r for r in rows if r["method"] == "none")
    assert none_row["mean_tone_preservation_rate"] is None


def test_format_markdown_contains_category_separators(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    md = evaluate_module.format_markdown(rows)
    assert "**baseline**" in md
    assert "**proposed**" in md
    assert "lacc" in md


def test_format_markdown_empty_rows(evaluate_module):
    assert "no results" in evaluate_module.format_markdown([])


def test_format_csv_round_trips_row_count(results_dir, evaluate_module):
    results = evaluate_module.load_results(results_dir)
    rows = evaluate_module.build_rows(results, context="registry")
    lines = [line for line in evaluate_module.format_csv(rows).strip().splitlines() if line]
    assert len(lines) == len(rows) + 1  # +1 header


# ============================================================================
# Dataset provenance (data/benchmark/)
# ============================================================================


def test_data_dir_resolves_to_data_benchmark():
    from scripts.checksum_datasets import DATA_DIR

    assert os.path.normpath(DATA_DIR).replace(os.sep, "/").endswith("data/benchmark")
    assert os.path.isdir(DATA_DIR)


def test_provenance_doc_exists():
    from scripts.checksum_datasets import DATA_DIR

    assert os.path.exists(os.path.join(DATA_DIR, "PROVENANCE.md"))


def test_checksums_manifest_matches_files_on_disk():
    from scripts.checksum_datasets import LOCALLY_GENERATED, MANIFEST_PATH, compute_manifest

    if not os.path.exists(MANIFEST_PATH):
        pytest.skip("CHECKSUMS.json not generated")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f)
    actual = compute_manifest()

    unregistered = set(actual) - set(recorded) - LOCALLY_GENERATED
    assert not unregistered, (
        f"dataset file(s) on disk but not in CHECKSUMS.json: {sorted(unregistered)} -- "
        "run `python scripts/checksum_datasets.py --write` and commit the update."
    )
    missing = set(recorded) - set(actual)
    assert not missing, f"CHECKSUMS.json records file(s) missing on disk: {sorted(missing)}"
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
    from scripts.checksum_datasets import DATA_DIR

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    for field in ("version", "date", "license"):
        assert field in meta, f"{filename} metadata missing required provenance field: {field}"
