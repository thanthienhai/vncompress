"""Tests for vncompress/config.py (unified experiment configuration)."""
import json
import random

import pytest

from vncompress.config import (
    ExperimentConfig,
    load_experiment_config,
    save_run_metadata,
    set_seed,
    snapshot_environment,
)


def test_default_config_has_expected_fields():
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.device == "cuda"
    assert config.compression_ratios == [2.0, 4.0, 8.0]
    assert config.split == "full"


def test_resolved_tokenizer_defaults_to_model():
    config = ExperimentConfig(model="foo/bar")
    assert config.resolved_tokenizer() == "foo/bar"


def test_resolved_tokenizer_explicit_override():
    config = ExperimentConfig(model="foo/bar", tokenizer="baz/qux")
    assert config.resolved_tokenizer() == "baz/qux"


def test_load_experiment_config_with_no_args_returns_defaults():
    config = load_experiment_config()
    assert config == ExperimentConfig()


def test_load_experiment_config_from_json_file(tmp_path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"seed": 7, "device": "cpu"}), encoding="utf-8")
    config = load_experiment_config(str(path))
    assert config.seed == 7
    assert config.device == "cpu"
    assert config.model == ExperimentConfig().model  # untouched fields keep defaults


def test_cli_overrides_take_precedence_over_config_file(tmp_path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"seed": 7, "device": "cpu"}), encoding="utf-8")
    config = load_experiment_config(str(path), cli_overrides={"device": "cuda"})
    assert config.seed == 7          # from file, no CLI override
    assert config.device == "cuda"   # CLI wins over file


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


def test_yaml_config_without_pyyaml_raises_or_loads(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("seed: 7\ndevice: cpu\n", encoding="utf-8")
    try:
        import yaml  # noqa: F401
        config = load_experiment_config(str(path))
        assert config.seed == 7
        assert config.device == "cpu"
    except ImportError:
        with pytest.raises(ImportError):
            load_experiment_config(str(path))


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
    assert "package_versions" in snap
    assert "torch" in snap["package_versions"]


def test_save_run_metadata_writes_both_files(tmp_path):
    config = ExperimentConfig(seed=99, output_dir=str(tmp_path))
    save_run_metadata(str(tmp_path), config)

    config_path = tmp_path / "config.json"
    env_path = tmp_path / "environment.json"
    assert config_path.exists()
    assert env_path.exists()

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["seed"] == 99

    saved_env = json.loads(env_path.read_text(encoding="utf-8"))
    assert "git_commit" in saved_env


class TestExperimentConfigValidation:
    """Guards against settings that used to crash mid-run or, worse, produce
    plausible-looking nonsense.

    Measured before the guard existed, with a 31-token input:
      ratio=0   -> ZeroDivisionError in 5 of 6 compressors, only after the
                   generation model had already been loaded.
      ratio=-2  -> no compressor raised, and they disagreed with each other
                   (random/selective/morphology_aware returned 4 tokens,
                   combined returned all 31).
    """

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

    def test_defaults_are_valid(self):
        cfg = ExperimentConfig()
        assert cfg.compression_ratios == [2.0, 4.0, 8.0]
        assert cfg.device == "cuda"
