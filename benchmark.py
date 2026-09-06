#!/usr/bin/env python3
"""benchmark.py -- VCC-Bench: Vietnamese Context Compression Benchmark.

    python benchmark.py --list-methods
    python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --demo
    python benchmark.py --config configs/benchmark.json
    python benchmark.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8
    python benchmark.py --ablation --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8

Ablation isolates LACC's three signals (perplexity/tone/morphology) as
config, not separate methods or scripts -- see docs/benchmark.md.

This script:
  1. Loads model + tokenizer (and, if given, the LACC scorer)
  2. Loads the VCC-Bench dataset (or falls back to a small demo set)
  3. Runs each configured method at each configured compression ratio
  4. Prints a comparison table and saves detailed results + a config/environment snapshot
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress.compression import METHODS, create_compressor
from vncompress.config import ExperimentConfig, load_experiment_config, save_run_metadata, set_seed
from vncompress.evaluation import VCCBench, VCCBenchConfig, VCCBenchSample, evaluate_compression
from vncompress.linguistics import get_tone_analyzer, is_vietnamese
from vncompress.models import load_model, load_scorer

ABLATION_METHODS = ['ppl_only', 'tone_only', 'morph_only', 'lacc']
ABLATION_KWARGS = {
    'ppl_only': dict(use_tone=False, use_morphology=False),
    'tone_only': dict(use_perplexity=False, use_morphology=False),
    'morph_only': dict(use_perplexity=False, use_tone=False),
    'lacc': dict(),
}

# Wave-2 arms (research/wave2_proposals.md). Each maps an arm name to a base
# registry method + constructor kwargs. LACC arms are given the loaded scorer;
# 'llmlingua_contrastive'/'encoder' are not. Select with e.g.
#   --methods none,llmlingua,llmlingua_contrastive,lacc_ppl_contrastive,lacc_ppl_morph
WAVE2_ARMS = {
    # E1/E11: LongLLMLingua question-conditioned (contrastive) perplexity baseline.
    'llmlingua_contrastive': ('llmlingua', dict(contrastive=True)),
    # E1: LACC using only query-conditioned perplexity (the highest-value arm).
    'lacc_ppl_contrastive': ('lacc', dict(use_tone=False, use_morphology=False, contrastive_ppl=True)),
    # E2: perplexity x morphology (multiplicative), tone off.
    'lacc_ppl_morph': ('lacc', dict(use_tone=False, morph_combine='multiply')),
    # E1+E2: query-conditioned perplexity x morphology.
    'lacc_cx_morph': ('lacc', dict(use_tone=False, morph_combine='multiply', contrastive_ppl=True)),
    # E5: sentence-level extractive selection.
    'lacc_sentence': ('lacc', dict(selection_unit='sentence', use_tone=False, contrastive_ppl=True)),
    # E7: class-proportional budget allocation.
    'lacc_classprop': ('lacc', dict(budget_mode='class_proportional', use_tone=False)),
    # E8: tone kept only for surface tasks.
    'lacc_tone_gated': ('lacc', dict(tone_task_gate=True)),
    # E6/E11: encoder token-classification compressor (LLMLingua-2 / PhoBERT-style).
    'encoder': ('encoder', dict()),
}

DEFAULT_OUTPUT_DIR = './results'
DEFAULT_ABLATION_OUTPUT_DIR = './results_ablation'


# ============================================================================
# Demo dataset: small Vietnamese long-context samples (used when no dataset
# file is found, and always for --demo)
# ============================================================================

VIETNAMESE_DEMO_SAMPLES = [
    VCCBenchSample(
        task='long_document_qa',
        context=(
            "Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
            "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
            "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. "
            "Điều 4 quy định nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường "
            "là quyền, nghĩa vụ và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng "
            "dân cư, hộ gia đình và cá nhân. Hoạt động bảo vệ môi trường phải được "
            "tiến hành thường xuyên, công khai, minh bạch; ưu tiên dự báo, phòng ngừa "
            "ô nhiễm, sự cố, suy thoái môi trường."
        ) * 3,
        query="Nguyên tắc bảo vệ môi trường được quy định như thế nào trong Luật?",
        reference_answer=(
            "Nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường là quyền và "
            "trách nhiệm của mọi tổ chức, cá nhân; hoạt động bảo vệ môi trường phải "
            "thường xuyên, công khai, minh bạch; ưu tiên phòng ngừa ô nhiễm; gắn kết "
            "với phát triển kinh tế và an sinh xã hội."
        ),
        context_length=0,
    ),
    VCCBenchSample(
        task='multi_turn_conversation',
        context=(
            "Người dùng: Chào bạn, tôi cần tư vấn về thủ tục đăng ký kinh doanh.\n"
            "Trợ lý: Chào anh/chị. Để đăng ký kinh doanh, anh/chị cần chuẩn bị những "
            "giấy tờ sau: đơn đăng ký, bản sao CMND/CCCD, và giấy tờ chứng minh "
            "địa điểm kinh doanh.\n"
            "Người dùng: Tôi muốn mở một cửa hàng bán đồ ăn nhanh thì cần thêm "
            "giấy tờ gì không?\n"
            "Trợ lý: Với ngành thực phẩm, anh/chị cần thêm giấy chứng nhận vệ sinh "
            "an toàn thực phẩm và giấy khám sức khỏe của chủ cơ sở."
        ) * 2,
        query="Để mở cửa hàng đồ ăn nhanh cần những giấy tờ gì?",
        reference_answer=(
            "Cần đơn đăng ký kinh doanh, CMND/CCCD, giấy chứng nhận địa điểm, "
            "giấy vệ sinh an toàn thực phẩm, và giấy khám sức khỏe."
        ),
        context_length=0,
    ),
    VCCBenchSample(
        task='needle_in_haystack',
        context=(
            "Công ty Cổ phần XYZ được thành lập vào ngày 15 tháng 3 năm 2010 "
            "tại Thành phố Hồ Chí Minh. Công ty hoạt động trong lĩnh vực công nghệ "
            "thông tin và viễn thông. Năm 2025, công ty đạt doanh thu 5000 tỷ đồng "
            "và lợi nhuận sau thuế là 850 tỷ đồng. MẬT KHẨU BÍ MẬT: VIETCOMPRESS2026 "
            "Công ty hiện có hơn 5000 nhân viên làm việc tại 10 chi nhánh trên "
            "toàn quốc và 3 văn phòng đại diện tại nước ngoài."
        ) * 5,
        query="Mật khẩu bí mật được đề cập trong văn bản là gì?",
        reference_answer="VIETCOMPRESS2026",
        context_length=0,
    ),
    VCCBenchSample(
        task='agent_tool_calling',
        context=(
            "Agent: Tôi cần tìm thông tin về thời tiết Hà Nội hôm nay.\n"
            "Tool: get_weather(location='Hà Nội', date='2026-06-28')\n"
            "Result: Nhiệt độ 35°C, độ ẩm 75%, có mưa rào vào chiều tối.\n"
            "User: Còn Đà Nẵng thì sao?\n"
            "Tool: get_weather(location='Đà Nẵng', date='2026-06-28')\n"
            "Result: Nhiệt độ 32°C, độ ẩm 80%, trời nắng.\n"
        ) * 2,
        query="Thời tiết Hà Nội và Đà Nẵng hôm nay như thế nào?",
        reference_answer=(
            "Hà Nội: 35°C, độ ẩm 75%, có mưa rào chiều tối. "
            "Đà Nẵng: 32°C, độ ẩm 80%, trời nắng."
        ),
        context_length=0,
    ),
    VCCBenchSample(
        task='cross_lingual',
        context=(
            "The Vietnamese economy has shown remarkable resilience in recent years. "
            "GDP growth reached 7.2% in 2025, driven by strong exports and foreign "
            "direct investment. Nền kinh tế Việt Nam đã thể hiện sức phục hồi đáng kể. "
            "Tăng trưởng GDP đạt 7.2% năm 2025, được thúc đẩy bởi xuất khẩu mạnh "
            "và đầu tư trực tiếp nước ngoài."
        ) * 2,
        query="Tăng trưởng GDP của Việt Nam năm 2025 là bao nhiêu?",
        reference_answer="Tăng trưởng GDP của Việt Nam năm 2025 là 7.2%.",
        context_length=0,
    ),
]


def _default_data_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'benchmark', 'vcc_bench_v1.json')


def _load_dataset(config: VCCBenchConfig, data_path, tokenizer, quick: bool) -> VCCBench:
    json_path = data_path or _default_data_path()
    if os.path.exists(json_path):
        print(f"\nLoading VCC-Bench dataset: {json_path}")
        return VCCBench.load_from_json(json_path, config)
    print(f"\n[WARN] Dataset not found at {json_path}, using demo samples")
    bench = VCCBench(config)
    samples = VIETNAMESE_DEMO_SAMPLES[:2] if quick else VIETNAMESE_DEMO_SAMPLES
    for sample in samples:
        sample.context_length = len(tokenizer.encode(sample.context))
    bench.add_samples(samples)
    return bench


def print_incremental_contribution(results: dict):
    """Print a table isolating how much each LACC signal adds over the others."""
    summary = results.get('summary', {})
    missing = [m for m in ABLATION_METHODS if m not in summary]
    if missing:
        print(f"\n[WARN] Missing methods in summary: {missing}; cannot compute incremental contribution table")
        return

    single_signal_scores = {m: summary[m]['harmonized_score'] for m in ('ppl_only', 'tone_only', 'morph_only')}
    combined_score = summary['lacc']['harmonized_score']
    best_single_name = max(single_signal_scores, key=single_signal_scores.get)
    best_single_score = single_signal_scores[best_single_name]

    print("\n" + "=" * 70 + "\nABLATION: Incremental Contribution of Each Signal\n" + "=" * 70)
    print(f"{'Signal':<15} {'Quality':>10} {'Efficiency':>10} {'Harmonized':>12}")
    print("-" * 50)
    for name in ABLATION_METHODS:
        scores = summary[name]
        print(f"{name:<15} {scores['avg_quality']:>10.4f} {scores['avg_efficiency']:>10.4f} {scores['harmonized_score']:>12.4f}")
    print("-" * 50)
    print(f"Best single-signal method: {best_single_name} ({best_single_score:.4f})")
    print(f"lacc vs. best single signal: {combined_score - best_single_score:+.4f}")
    print(f"lacc vs. ppl_only:           {combined_score - single_signal_scores['ppl_only']:+.4f}")
    print("=" * 70)


def run_benchmark(
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    device: str = 'cuda',
    methods: list = None,
    ratios: list = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    quick: bool = False,
    data_path: str = None,
    exp_config: 'ExperimentConfig' = None,
    scorer_adapter_dir: str = None,
    tone_probe_path: str = None,
    encoder_path: str = None,
    encoder_id: str = None,
    ablation: bool = False,
):
    """Run VCC-Bench evaluation (or, with ablation=True, the signal-isolation
    ablation study). Writes config.json + environment.json into `output_dir`
    before the run starts, if `exp_config` is given -- see docs/benchmark.md."""
    if exp_config is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_run_metadata(output_dir, exp_config)

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device, dtype='float16')
    print(f"Model loaded. Vocab size: {len(tokenizer)}, "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    if exp_config is not None:
        # Seed AFTER the model is on its device -- see docs/benchmark.md
        # (early torch.cuda.manual_seed_all() has corrupted the CUDA
        # allocator on this project's Windows/CUDA dev machine).
        set_seed(exp_config.seed)

    scorer = load_scorer(scorer_adapter_dir, tone_probe_path=tone_probe_path, device=device) if scorer_adapter_dir else None

    method_names = ABLATION_METHODS if ablation else (methods or list(dict.fromkeys(['none', 'random', 'llmlingua', 'lacc'])))
    config = VCCBenchConfig(
        methods=method_names,
        compression_ratios=ratios or ([2.0] if quick else [2.0, 4.0, 8.0]),
        output_dir=output_dir, device=device,
        max_new_tokens=128 if quick else 256,
    )
    bench = _load_dataset(config, data_path, tokenizer, quick)

    print("\nBenchmark Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Methods: {config.methods}")
    print(f"  Ratios: {[f'{r}x' for r in config.compression_ratios]}")
    print(f"  Samples: {bench.total_samples}")

    tone_analyzer = get_tone_analyzer()
    all_samples = [s for task_samples in bench.samples.values() for s in task_samples]
    vi_samples = [s for s in all_samples if is_vietnamese(s.context[:500])]
    if vi_samples:
        sample_tokens = [tokenizer.decode([tid]).strip() for tid in tokenizer.encode(vi_samples[0].context)[:100]]
        tone_stats = tone_analyzer.analyze_tokens(sample_tokens)
        avg_weight = sum(t.preservation_weight for t in tone_stats) / max(len(tone_stats), 1)
        print("\n  Tone Analysis (first 100 tokens):")
        print(f"    Avg tone preservation weight: {avg_weight:.3f}")
        print(f"    Tone-bearing tokens: {sum(1 for t in tone_stats if t.tones_present)}/{len(tone_stats)}")

    def make_compressor(method_name: str):
        if ablation:
            return create_compressor('lacc', tokenizer, model, config=None, device=device, scorer=scorer, **ABLATION_KWARGS[method_name])
        if method_name in WAVE2_ARMS:
            base, kw = WAVE2_ARMS[method_name]
            if base == 'lacc':
                extra = dict(scorer=scorer)
            elif base == 'encoder':
                # E6: point the arm at a fine-tuned keep/drop checkpoint
                # (--encoder-path, from scripts/train_encoder_compressor.py) or,
                # failing that, a raw encoder id (--encoder-id) to smoke-test the
                # wiring. Without either, EncoderClassifierCompressor.compress()
                # raises rather than silently producing garbage.
                if not (encoder_path or encoder_id):
                    raise ValueError(
                        "The 'encoder' arm needs a checkpoint: pass --encoder-path "
                        "(a dir from scripts/train_encoder_compressor.py) or "
                        "--encoder-id (a HF encoder id, e.g. vinai/phobert-base)."
                    )
                extra = dict(encoder_path=encoder_path, encoder_id=encoder_id)
            else:
                extra = {}
            return create_compressor(base, tokenizer, model, config=None, device=device, **extra, **kw)
        if method_name == 'lacc':
            return create_compressor('lacc', tokenizer, model, config=None, device=device, scorer=scorer)
        return create_compressor(method_name, tokenizer, model, config=None, device=device)

    results = bench.evaluate(compressor_fn=make_compressor, model=model, tokenizer=tokenizer)
    bench.print_summary(results)
    if ablation:
        print_incremental_contribution(results)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "ablation_results.json" if ablation else "vcc_bench_results.json")
    serializable = {k: v for k, v in results.items() if isinstance(v, dict)}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {results_path}")

    report = bench.generate_report(results)
    report_path = os.path.join(output_dir, "ablation_report.md" if ablation else "vcc_bench_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    return results


def quick_demo(model_name: str = 'Qwen/Qwen2.5-7B-Instruct', device: str = 'cuda'):
    """Run every method on one hardcoded sample and print per-method detail."""
    print("=" * 60 + "\nVNCOMPRESS -- Quick Demo\n" + "=" * 60)
    print(f"\nLoading {model_name}...")
    model, tokenizer = load_model(model_name, device=device, dtype='float16')

    sample = VIETNAMESE_DEMO_SAMPLES[0]
    print(f"\nOriginal context: {len(tokenizer.encode(sample.context))} tokens")
    print(f"Query: {sample.query}")
    print(f"Reference: {sample.reference_answer}\n")

    for method_name in ['none', 'llmlingua', 'lacc']:
        print(f"\n--- {method_name} ---")
        compressor = create_compressor(method_name, tokenizer, model, device=device)
        metric = evaluate_compression(
            compressor, model, tokenizer, input_text=sample.context,
            query=sample.query, reference=sample.reference_answer, ratio=4.0,
        )
        print(f"  Compression: {metric.compression_ratio:.1f}x")
        print(f"  Token savings: {metric.token_savings_pct:.1f}%")
        print(f"  Processing: {metric.processing_time_ms:.1f}ms")
        print(f"  ROUGE-L F1: {metric.rouge_l_f1:.4f}" if metric.rouge_l_f1 is not None else "  ROUGE-L: N/A")
        print(f"  Quality: {metric.quality_score:.4f}")
        print(f"  Efficiency: {metric.efficiency_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='VCC-Bench: Vietnamese Context Compression Benchmark')
    parser.add_argument('--config', default=None, help='ExperimentConfig JSON. CLI flags below override it.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model', default=None, help='Default: Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--methods', default=None, help=f"Comma-separated method names, from {list(METHODS)}")
    parser.add_argument('--ratios', default=None, help='Comma-separated compression ratios (default: 2,4,8)')
    parser.add_argument('--output-dir', default=None, help=f'Default: {DEFAULT_OUTPUT_DIR} ({DEFAULT_ABLATION_OUTPUT_DIR} for --ablation)')
    parser.add_argument('--quick', action='store_true', help='Fewer ratios/samples')
    parser.add_argument('--demo', action='store_true', help='Single-sample per-method detail, not a comparable run')
    parser.add_argument('--ablation', action='store_true', help='Isolate perplexity/tone/morphology signals (see docs/benchmark.md)')
    parser.add_argument('--list-methods', action='store_true')
    parser.add_argument('--data-path', default=None, help='Default: data/benchmark/vcc_bench_v1.json')
    parser.add_argument('--scorer-adapter-dir', default=None,
                         help="LACC scorer: a LoRA adapter dir from 'train.py --mode slm' (e.g. models/slm/final) "
                              "or a HuggingFace model id. Enables the lightweight-tier perplexity signal.")
    parser.add_argument('--tone-probe-path', default=None,
                         help="Trained tone probe (e.g. models/slm/tone_probe.pt), paired with --scorer-adapter-dir. "
                              "Enables LACC's trained-tone-probe signal instead of the rule-based one.")
    parser.add_argument('--encoder-path', default=None,
                         help="Fine-tuned keep/drop encoder checkpoint dir (from scripts/train_encoder_compressor.py), "
                              "for the 'encoder' arm (wave-2 E6). Takes precedence over --encoder-id.")
    parser.add_argument('--encoder-id', default=None,
                         help="Raw encoder id for the 'encoder' arm when no fine-tuned checkpoint is available "
                              "(e.g. vinai/phobert-base) -- smoke-tests the wiring only.")
    args = parser.parse_args()

    if args.list_methods:
        print("Available compression methods:")
        for name, cls in METHODS.items():
            print(f"  {name:<22} -> {cls.__name__}")
        print("  encoder                -> EncoderClassifierCompressor (lazy; wave-2 E6)")
        print(f"\nAblation arms (--ablation): {ABLATION_METHODS}")
        print("\nWave-2 arms (research/wave2_proposals.md), selectable via --methods:")
        for name, (base, kw) in WAVE2_ARMS.items():
            print(f"  {name:<22} -> {base} {kw}")
        return

    methods = args.methods.split(',') if args.methods else None
    ratios = None
    if args.ratios:
        try:
            ratios = [float(r) for r in args.ratios.split(',')]
        except ValueError:
            parser.error(f"Invalid compression ratios: '{args.ratios}'. Use comma-separated numbers, e.g. --ratios 2,4,8")
    if args.quick and ratios is None:
        ratios = [2.0]

    exp_config = load_experiment_config(
        config_path=args.config,
        cli_overrides={
            'seed': args.seed, 'model': args.model, 'device': args.device,
            'methods': methods, 'compression_ratios': ratios,
            'output_dir': args.output_dir, 'data_path': args.data_path,
        },
    )
    if args.output_dir is None and args.config is None and args.ablation:
        exp_config.output_dir = DEFAULT_ABLATION_OUTPUT_DIR

    if args.demo:
        quick_demo(exp_config.model, exp_config.device)
        return

    run_benchmark(
        model_name=exp_config.model, device=exp_config.device, methods=exp_config.methods,
        ratios=exp_config.compression_ratios, output_dir=exp_config.output_dir, quick=args.quick,
        data_path=exp_config.data_path, exp_config=exp_config,
        scorer_adapter_dir=args.scorer_adapter_dir, tone_probe_path=args.tone_probe_path,
        encoder_path=args.encoder_path, encoder_id=args.encoder_id,
        ablation=args.ablation,
    )


if __name__ == '__main__':
    main()
