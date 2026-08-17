#!/usr/bin/env python3
"""
Ablation Study — Isolating Each LACC Signal
=============================================
Runs VCC-Bench with the perplexity, tone, and morphology signals isolated
one at a time, plus the full combined method, so each signal's incremental
contribution to the final quality/efficiency trade-off can be measured and
reported in the paper.

Ablation arms:
  - ppl_only    : LLMLingua-style perplexity importance only (no tone/morph
                  multiplier at all) — the pre-existing 'llmlingua' baseline.
  - tone_only   : CombinedCompressor with model=None (neutralizes the
                  perplexity term to a uniform base score, see
                  ToneAwareCompressor._compute_base_scores) and tone_weight=1.0
                  (morphology multiplier weighted out of the blend).
  - morph_only  : Same trick, tone_weight=0.0 (tone multiplier weighted out).
  - combined    : CombinedCompressor with the real model (perplexity signal
                  active) and tone_weight=0.5 — the full method.

Only the token-selection signal changes between arms; downstream generation
always uses the same real model, so ROUGE-L/BLEU/EM differences are
attributable to the compression signal alone.

Usage:
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --quick
"""

import argparse
import json
import os
import sys

# Add parent to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress.config import (
    ExperimentConfig, load_experiment_config, save_run_metadata, set_seed,
)
from vncompress.evaluation import VCCBench, VCCBenchConfig
from vncompress.compressors import (
    CombinedCompressor,
    LLMLinguaCompressor,
    CompressionConfig,
)

DEFAULT_OUTPUT_DIR = './results_ablation'

ABLATION_METHODS = ['ppl_only', 'tone_only', 'morph_only', 'combined']


def make_ablation_compressor(method_name: str, tokenizer, model, config, device: str = 'cuda'):
    """
    Build the compressor for one ablation arm.

    tone_only / morph_only pass model=None into CombinedCompressor so its
    internal ToneAwareCompressor._compute_base_scores() returns a uniform
    base score instead of a real perplexity score (see tone_aware.py:
    `if self.base_method == 'uniform' or self.model is None: return
    torch.ones(n)`) — this isolates the tone/morphology multiplier as the
    *only* signal driving token selection, with no GPU needed for that arm.
    """
    if method_name == 'ppl_only':
        return LLMLinguaCompressor(tokenizer, model, config=config, device=device)
    if method_name == 'tone_only':
        return CombinedCompressor(tokenizer, None, config, device, tone_weight=1.0)
    if method_name == 'morph_only':
        return CombinedCompressor(tokenizer, None, config, device, tone_weight=0.0)
    if method_name == 'combined':
        return CombinedCompressor(tokenizer, model, config, device, tone_weight=0.5)
    raise ValueError(f"Unknown ablation method: {method_name!r}. Expected one of {ABLATION_METHODS}")


def print_incremental_contribution(results: dict):
    """Print a table isolating how much each signal adds over the others."""
    summary = results.get('summary', {})
    missing = [m for m in ABLATION_METHODS if m not in summary]
    if missing:
        print(f"\n[WARN] Missing methods in summary: {missing}; "
              "cannot compute incremental contribution table")
        return

    single_signal_scores = {
        m: summary[m]['harmonized_score']
        for m in ('ppl_only', 'tone_only', 'morph_only')
    }
    combined_score = summary['combined']['harmonized_score']
    best_single_name = max(single_signal_scores, key=single_signal_scores.get)
    best_single_score = single_signal_scores[best_single_name]

    print("\n" + "=" * 70)
    print("ABLATION: Incremental Contribution of Each Signal")
    print("=" * 70)
    print(f"{'Signal':<15} {'Quality':>10} {'Efficiency':>10} {'Harmonized':>12}")
    print("-" * 50)
    for name in ABLATION_METHODS:
        scores = summary[name]
        print(f"{name:<15} {scores['avg_quality']:>10.4f} "
              f"{scores['avg_efficiency']:>10.4f} {scores['harmonized_score']:>12.4f}")
    print("-" * 50)
    print(f"Best single-signal method: {best_single_name} ({best_single_score:.4f})")
    print(f"Combined vs. best single signal: {combined_score - best_single_score:+.4f}")
    print(f"Combined vs. ppl_only baseline:  {combined_score - single_signal_scores['ppl_only']:+.4f}")
    print("=" * 70)


def run_ablation(
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    device: str = 'cuda',
    ratios: list = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    quick: bool = False,
    data_path: str = None,
    exp_config: 'ExperimentConfig' = None,
):
    """Run the ablation study end-to-end and print/save a comparison report.

    If `exp_config` is given, its seed is applied and a config.json +
    environment.json snapshot is written to `output_dir` before the run
    starts -- see vncompress/config.py and docs/benchmark.md.
    """
    if exp_config is not None:
        set_seed(exp_config.seed)
        os.makedirs(output_dir, exist_ok=True)
        save_run_metadata(output_dir, exp_config)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_benchmark import VIETNAMESE_DEMO_SAMPLES

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {'trust_remote_code': True, 'torch_dtype': torch.float16}
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    config = VCCBenchConfig(
        methods=ABLATION_METHODS,
        compression_ratios=ratios or ([2.0] if quick else [2.0, 4.0, 8.0]),
        output_dir=output_dir,
        device=device,
        max_new_tokens=128 if quick else 256,
    )

    default_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'vcc_bench_data', 'vcc_bench_v1.json'
    )
    json_path = data_path or default_data

    if os.path.exists(json_path):
        print(f"\nLoading VCC-Bench dataset: {json_path}")
        bench = VCCBench.load_from_json(json_path, config)
    else:
        print(f"\n[WARN] Dataset not found at {json_path}, using demo samples")
        bench = VCCBench(config)
        samples = VIETNAMESE_DEMO_SAMPLES[:2] if quick else VIETNAMESE_DEMO_SAMPLES
        for sample in samples:
            sample.context_length = len(tokenizer.encode(sample.context))
        bench.add_samples(samples)

    print("\nAblation Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Arms: {config.methods}")
    print(f"  Ratios: {[f'{r}x' for r in config.compression_ratios]}")
    print(f"  Samples: {bench.total_samples}")

    def make_compressor(method_name: str):
        return make_ablation_compressor(
            method_name, tokenizer, model, CompressionConfig(), device
        )

    results = bench.evaluate(
        compressor_fn=make_compressor,
        model=model,
        tokenizer=tokenizer,
    )

    bench.print_summary(results)
    print_incremental_contribution(results)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "ablation_results.json")
    serializable = {k: v for k, v in results.items() if isinstance(v, dict)}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {results_path}")

    report = bench.generate_report(results)
    report_path = os.path.join(output_dir, "ablation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='LACC Ablation Study: isolate tone/morph/perplexity signals'
    )
    # None defaults for ExperimentConfig-owned fields -- see run_benchmark.py's
    # main() and vncompress/config.py for the CLI/config-file precedence rule.
    parser.add_argument('--config', type=str, default=None,
                         help='Path to a JSON (or YAML, if PyYAML is installed) ExperimentConfig file.')
    parser.add_argument('--seed', type=int, default=None,
                         help='Random seed for all stochastic components (default: 42)')
    parser.add_argument('--model', type=str, default=None,
                         help='Model name (HuggingFace). Default: Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--device', type=str, default=None,
                         choices=['cuda', 'cpu', 'mps'], help='Device to run on. Default: cuda')
    parser.add_argument('--ratios', type=str, default=None,
                         help='Comma-separated compression ratios (default: 2,4,8)')
    parser.add_argument('--output-dir', type=str, default=None,
                         help=f'Output directory for results. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--quick', action='store_true',
                         help='Run quick evaluation with fewer ratios and samples')
    parser.add_argument('--data-path', type=str, default=None,
                         help='Path to VCC-Bench JSON dataset (default: vcc_bench_data/vcc_bench_v1.json)')

    args = parser.parse_args()

    ratios = None
    if args.ratios:
        try:
            ratios = [float(r) for r in args.ratios.split(',')]
        except ValueError:
            parser.error(f"Invalid compression ratios: '{args.ratios}'. "
                         "Use comma-separated numbers, e.g. --ratios 2,4,8")
    if args.quick and ratios is None:
        ratios = [2.0]

    exp_config = load_experiment_config(
        config_path=args.config,
        cli_overrides={
            'seed': args.seed,
            'model': args.model,
            'device': args.device,
            'compression_ratios': ratios,
            'output_dir': args.output_dir,
            'data_path': args.data_path,
        },
    )
    # run_ablation.py's own default output dir differs from
    # ExperimentConfig's ('./results') -- only apply it when nothing
    # explicit (CLI or config file) set output_dir.
    if args.output_dir is None and args.config is None:
        exp_config.output_dir = DEFAULT_OUTPUT_DIR
    set_seed(exp_config.seed)

    run_ablation(
        model_name=exp_config.model,
        device=exp_config.device,
        ratios=exp_config.compression_ratios,
        output_dir=exp_config.output_dir,
        quick=args.quick,
        data_path=exp_config.data_path,
        exp_config=exp_config,
    )


if __name__ == '__main__':
    main()
