#!/usr/bin/env python3
"""
run_colab.py — Optimized Benchmark for Colab T4 / Kaggle P100 / 2xT4
=====================================================================

Runs VCC-Bench on limited hardware (16GB VRAM) using:
  - INT4 quantization (7B model in ~5GB)
  - No-model compressors (0MB VRAM for compression)
  - Tiny model scoring (<500M params in <1GB VRAM)
  - Short context (1K-2K tokens)
  - Memory-efficient generation

MODES (auto-selected based on available VRAM):

  Mode 1 — FULL (>=24GB):        INT4 7B + tiny scorer + all compressors
  Mode 2 — LIGHTWEIGHT (16GB):   INT4 7B + no-model compressors
  Mode 3 — MINIMAL (10-15GB):    INT4 3B + no-model compressors
  Mode 4 — CPU ONLY (<10GB):     No GPU compression, CPU-only eval

Usage:

  # Auto-detect best mode
  python run_colab.py --auto

  # Force specific mode
  python run_colab.py --mode lightweight
  python run_colab.py --mode no_model   # CPU-only, 0 VRAM

  # Custom model + ratio
  python run_colab.py --model Qwen/Qwen2.5-3B-Instruct --ratio 4

  # Skip generation (just measure compression)
  python run_colab.py --skip-generation

VRAM Budget Analysis:
  ┌─────────────────────────────────────────────────────┐
  │ Component              │ INT4 7B │ INT4 3B │ None   │
  ├─────────────────────────────────────────────────────┤
  │ Model weights          │  4.5 GB │  2.0 GB │ 0      │
  │ KV Cache (2K ctx)      │  0.8 GB │  0.4 GB │ 0      │
  │ Tiny scorer (0.5B)     │  0.5 GB │  0.5 GB │ 0      │
  │ PyTorch overhead       │  1.5 GB │  1.0 GB │ 0.2 GB │
  │ TOTAL                  │  7.3 GB │  3.9 GB │ 0.2 GB │
  │ T4/P100 available      │ 15.0 GB │ 15.0 GB │ 15.0 GB│
  │ FREE                   │  7.7 GB │ 11.1 GB │ 14.8 GB│
  └─────────────────────────────────────────────────────┘

Tone/Morphology analysis always runs on CPU (0 VRAM) regardless of mode.
"""

import argparse
import json
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress.tone_aware import VietnameseToneAnalyzer, is_vietnamese, get_tone_analyzer
from vncompress.morphology import MorphologyAnalyzer, get_morphology_analyzer
from vncompress.compressors.no_model import (
    NoModelToneCompressor,
    NoModelMorphCompressor,
    NoModelCombinedCompressor,
    NoModelBaselineCompressor,
    evaluate_no_model,
)


# ============================================================================
# Lightweight Benchmark (no torch needed for compression)
# ============================================================================

def benchmark_no_model_compressors(
    texts: list,
    tokenizer,
    target_ratios: list = None,
) -> dict:
    """
    Benchmark only no-model compressors. No GPU, no torch model needed.
    """
    if target_ratios is None:
        target_ratios = [2.0, 4.0, 6.0, 8.0]
    
    compressors = {
        'baseline_first':   NoModelBaselineCompressor(tokenizer, mode='first'),
        'baseline_random':  NoModelBaselineCompressor(tokenizer, mode='random'),
        'baseline_longest': NoModelBaselineCompressor(tokenizer, mode='word_length'),
        'tone_only':        NoModelToneCompressor(tokenizer),
        'morph_only':       NoModelMorphCompressor(tokenizer),
        'combined':         NoModelCombinedCompressor(tokenizer),
    }
    
    results = {}
    
    for name, comp in compressors.items():
        for ratio in target_ratios:
            key = f"{name}@{ratio}x"
            ratio_results = []
            
            for text in texts:
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                result = comp.compress(input_ids, target_ratio=ratio)
                ratio_results.append({
                    'original_length': result.original_length,
                    'compressed_length': result.compressed_length,
                    'compression_ratio': result.compression_ratio,
                    'token_savings_pct': result.token_savings_pct,
                    'processing_time_ms': result.processing_time_ms,
                })
            
            # Aggregate
            avg_cr = sum(r['compression_ratio'] for r in ratio_results) / len(ratio_results)
            avg_ts = sum(r['token_savings_pct'] for r in ratio_results) / len(ratio_results)
            avg_time = sum(r['processing_time_ms'] for r in ratio_results) / len(ratio_results)
            
            results[key] = {
                'avg_compression_ratio': round(avg_cr, 2),
                'avg_token_savings_pct': round(avg_ts, 1),
                'avg_processing_time_ms': round(avg_time, 2),
                'num_samples': len(ratio_results),
            }
    
    return results


def benchmark_with_generation(
    texts: list,
    queries: list,
    references: list,
    tokenizer,
    model,
    target_ratios: list = None,
    max_new_tokens: int = 128,
) -> dict:
    """
    Full benchmark with generation quality evaluation.
    
    Requires: GPU with loaded model.
    """
    if target_ratios is None:
        target_ratios = [4.0]
    
    import torch
    from vncompress.evaluation.metrics import (
        compute_rouge_l, compute_bleu, compute_exact_match,
    )
    
    compressors = {
        'no_compress':  NoModelBaselineCompressor(tokenizer, mode='first'),
        'random':       NoModelBaselineCompressor(tokenizer, mode='random'),
        'tone_only':    NoModelToneCompressor(tokenizer),
        'morph_only':   NoModelMorphCompressor(tokenizer),
        'combined':     NoModelCombinedCompressor(tokenizer),
    }
    
    results = {}
    
    for name, comp in compressors.items():
        for ratio in target_ratios:
            key = f"{name}@{ratio}x"
            predictions, refs = [], []
            total_time = 0
            
            for text, query, ref in zip(texts, queries, references):
                # Compress
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                compress_result = comp.compress(input_ids, target_ratio=ratio)
                
                # Build prompt: compressed context + query
                query_ids = tokenizer.encode(query, add_special_tokens=False)
                full_input = compress_result.compressed_ids + query_ids
                
                # Generate
                input_tensor = torch.tensor([full_input], device=model.device)
                
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                total_time += (time.time() - start) * 1000
                
                generated = outputs[0][len(full_input):].cpu()
                pred_text = tokenizer.decode(generated, skip_special_tokens=True)
                
                predictions.append(pred_text)
                refs.append(ref)
            
            # Metrics
            rouge = compute_rouge_l(predictions, refs)
            bleu = compute_bleu(predictions, refs)
            em = compute_exact_match(predictions, refs)
            avg_time = total_time / len(texts)
            
            results[key] = {
                'rouge_l_f1': round(rouge.get('rougeL_f1', 0), 4),
                'bleu': round(bleu, 4),
                'exact_match': round(em, 4),
                'avg_generation_time_ms': round(avg_time, 1),
                'num_samples': len(predictions),
            }
    
    return results


# ============================================================================
# Main Runner
# ============================================================================

DEMO_TEXTS = [
    # Vietnamese legal
    (
        ("Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
         "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
         "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. Điều 4 quy định "
         "nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường là quyền, nghĩa vụ "
         "và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình và "
         "cá nhân. Hoạt động bảo vệ môi trường phải được tiến hành thường xuyên, "
         "công khai, minh bạch; ưu tiên dự báo, phòng ngừa ô nhiễm, sự cố, suy thoái "
         "môi trường. Bảo vệ môi trường gắn kết hài hòa với phát triển kinh tế.") * 2,
        "Nguyên tắc bảo vệ môi trường được quy định như thế nào?",
        "Nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường là quyền và trách nhiệm "
        "của mọi tổ chức, cá nhân; hoạt động phải thường xuyên, công khai, minh bạch; "
        "ưu tiên phòng ngừa ô nhiễm; gắn kết với phát triển kinh tế.",
    ),
    # Vietnamese news
    (
        ("Thị trường chứng khoán Việt Nam đã có phiên giao dịch tích cực vào ngày "
         "hôm nay khi chỉ số VN-Index tăng 12 điểm, đạt mức 1280 điểm. Khối lượng "
         "giao dịch đạt hơn 1 tỷ cổ phiếu với tổng giá trị giao dịch hơn 25 nghìn "
         "tỷ đồng. Nhóm cổ phiếu ngân hàng và bất động sản dẫn đầu đà tăng trưởng. "
         "Các chuyên gia nhận định thị trường sẽ tiếp tục xu hướng tích cực trong "
         "những phiên tới nhờ vào dòng tiền từ nhà đầu tư nước ngoài và kết quả "
         "kinh doanh quý 2 khả quan của các doanh nghiệp niêm yết.") * 2,
        "VN-Index hôm nay tăng bao nhiêu điểm và đạt mức nào?",
        "VN-Index tăng 12 điểm, đạt mức 1280 điểm.",
    ),
    # Vietnamese conversation
    (
        ("Người dùng: Chào bạn, thời tiết Hà Nội hôm nay thế nào?\n"
         "Trợ lý: Hà Nội hôm nay nắng nhẹ, nhiệt độ 28-35°C, độ ẩm 70%.\n"
         "Người dùng: Có mưa không?\n"
         "Trợ lý: Chiều tối có thể có mưa rào. Bạn nên mang ô.\n"
         "Người dùng: Cảm ơn bạn. Đà Nẵng thì sao?\n"
         "Trợ lý: Đà Nẵng nắng đẹp, 30-36°C, rất thích hợp đi biển.\n") * 2,
        "Thời tiết Hà Nội và Đà Nẵng hôm nay như thế nào?",
        "Hà Nội nắng nhẹ 28-35°C, chiều tối có mưa rào. Đà Nẵng nắng đẹp 30-36°C.",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description='VNCOMPRESS - Optimized for Colab/Kaggle T4/P100'
    )
    
    # Mode
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect best mode for current hardware')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'full', 'lightweight', 'no_model', 'cpu_only'],
                       help='Running mode')
    
    # Model (for modes that need GPU)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--tiny-scorer', type=str, default='smollm2-135m',
                       help='Tiny model for scoring (SMALL = fits in VRAM)')
    
    # Compression
    parser.add_argument('--ratio', type=float, default=4.0,
                       help='Target compression ratio')
    parser.add_argument('--ratios', type=str, default='2,4,6,8',
                       help='Comma-separated ratios for benchmark')
    
    # Generation
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip LLM generation (only measure compression)')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                       help='Max tokens to generate')
    
    # Output
    parser.add_argument('--output', type=str, default='./results/colab_results.json')
    
    # Tone/Morph params
    parser.add_argument('--tone-weight', type=float, default=0.5,
                       help='Tone vs morphology blend weight (0-1)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Tone base importance')
    parser.add_argument('--gamma', type=float, default=0.4,
                       help='Tone contrast amplification')
    
    args = parser.parse_args()
    
    # Parse ratios
    target_ratios = [float(r) for r in args.ratios.split(',')]
    
    # ================================================================
    # STEP 1: Detect hardware
    # ================================================================
    from vncompress.utils.lightweight import detect_gpu, print_vram_status
    
    hw = detect_gpu()
    
    print("=" * 70)
    print("  VNCOMPRESS — Colab/Kaggle Optimized Benchmark")
    print("=" * 70)
    print(f"  GPU: {hw['gpu_name']}")
    print(f"  VRAM: {hw['vram_gb']:.0f} GB")
    print(f"  Environment: {'Colab' if hw['colab_environment'] else 'Kaggle' if hw['kaggle_environment'] else 'Local'}")
    print(f"  CUDA: {'Yes' if hw['has_cuda'] else 'No'}")
    print("-" * 70)
    
    # Auto-detect mode
    if args.auto or args.mode == 'auto':
        if not hw['has_cuda']:
            args.mode = 'cpu_only'
        elif hw['vram_gb'] >= 20:
            args.mode = 'full'
        elif hw['vram_gb'] >= 14:
            args.mode = 'lightweight'
        else:
            args.mode = 'no_model'
        print(f"  Auto-selected mode: {args.mode}")
    
    # ================================================================
    # STEP 2: Load tokenizer (always needed)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 1: Loading tokenizer")
    print(f"{'='*70}")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {len(tokenizer)} vocab size")
    
    # ================================================================
    # STEP 3: Tone + Morphology Analysis (CPU, 0 VRAM)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 2: Tone & Morphology Analysis (0 VRAM)")
    print(f"{'='*70}")
    
    tone_analyzer = get_tone_analyzer(
        alpha=args.alpha, gamma=args.gamma
    )
    morph_analyzer = get_morphology_analyzer()
    
    # Show analysis on sample
    sample_text = DEMO_TEXTS[0][0][:200]
    sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    
    tokens = []
    for tid in sample_ids[:20]:
        t = tokenizer.decode([tid])
        t = t.replace('\u2581', ' ').replace('\u0130', ' ').strip()
        tokens.append(t)
    
    tone_infos = tone_analyzer.analyze_tokens(tokens)
    word_infos = morph_analyzer.classify_batch(tokens)
    
    print(f"\n  {'Token':<15} {'Tone':>8} {'Weight':>8} {'Class':<12}")
    print(f"  {'-'*45}")
    for ti, wi in zip(tone_infos[:15], word_infos[:15]):
        tone_name = ti.dominant_tone or 'ngang'
        print(f"  {ti.token[:14]:<15} {tone_name:>8} {ti.preservation_weight:>8.3f} {wi.word_class.value:<12}")
    
    print_vram_status("  ")
    
    # ================================================================
    # STEP 4: No-Model Compression Benchmark (0 VRAM)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 3: No-Model Compression Benchmark (0 VRAM)")
    print(f"{'='*70}")
    
    texts = [t[0] for t in DEMO_TEXTS]
    
    compression_results = benchmark_no_model_compressors(
        texts, tokenizer, target_ratios
    )
    
    print(f"\n  {'Method':<25} {'Ratio':>6} {'CR':>7} {'Saved':>8} {'Time':>8}")
    print(f"  {'-'*60}")
    for key, stats in sorted(compression_results.items()):
        parts = key.split('@')
        method = parts[0]
        ratio = float(parts[1].replace('x', ''))
        print(f"  {method:<25} {ratio:>5.0f}x {stats['avg_compression_ratio']:>6.1f}x "
              f"{stats['avg_token_savings_pct']:>7.0f}% {stats['avg_processing_time_ms']:>7.1f}ms")
    
    print_vram_status("  ")
    
    # ================================================================
    # STEP 5: Generation (GPU needed)
    # ================================================================
    all_results = {
        'hardware': hw,
        'mode': args.mode,
        'model': args.model,
        'compression': compression_results,
    }

    if args.skip_generation or args.mode == 'cpu_only':
        print(f"\n{'='*70}")
        print(f"  STEP 4: SKIPPED (--skip-generation / cpu_only)")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"  STEP 4: Loading model for generation")
        print(f"{'='*70}")
        
        try:
            from vncompress.utils.lightweight import load_model_4bit, clear_gpu_memory
            
            # Load INT4 model
            model, _ = load_model_4bit(args.model)
            model.eval()
            
            print_vram_status("  After model load: ")
            
            # Run generation benchmark
            print(f"\n  Generating with compressed contexts...")
            
            gen_results = benchmark_with_generation(
                texts=[t[0] for t in DEMO_TEXTS],
                queries=[t[1] for t in DEMO_TEXTS],
                references=[t[2] for t in DEMO_TEXTS],
                tokenizer=tokenizer,
                model=model,
                target_ratios=[args.ratio],
                max_new_tokens=args.max_new_tokens,
            )
            
            print(f"\n  {'Method':<25} {'Ratio':>6} {'ROUGE-L':>9} {'BLEU':>7} {'EM':>6} {'Time':>8}")
            print(f"  {'-'*65}")
            for key, stats in sorted(gen_results.items()):
                parts = key.split('@')
                method = parts[0]
                ratio = float(parts[1].replace('x', ''))
                print(f"  {method:<25} {ratio:>5.0f}x {stats['rouge_l_f1']:>8.4f} "
                      f"{stats['bleu']:>6.4f} {stats['exact_match']:>5.3f} "
                      f"{stats['avg_generation_time_ms']:>7.0f}ms")
            
            # Save combined results
            all_results = {
                'hardware': hw,
                'mode': args.mode,
                'model': args.model,
                'compression': compression_results,
                'generation': gen_results,
            }
            
            clear_gpu_memory()
            
        except Exception as e:
            print(f"\n  ⚠ Generation failed: {e}")
            print(f"  Continuing with compression-only results...")
            all_results = {
                'hardware': hw,
                'mode': args.mode,
                'model': args.model,
                'compression': compression_results,
                'generation_error': str(e),
            }
    
    # ================================================================
    # Save results
    # ================================================================
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"  Results saved to: {args.output}")
    print(f"  Mode used: {args.mode}")
    print(f"  Hardware: {hw['gpu_name']} ({hw['vram_gb']:.0f} GB)")
    print(f"{'='*70}")
    
    # ================================================================
    # Quick recommendations
    # ================================================================
    print(f"\n  💡 Recommendations for your hardware ({hw['gpu_name']}, {hw['vram_gb']:.0f}GB):")
    
    if args.mode == 'cpu_only':
        print(f"     ✓ Use --skip-generation for compression-only analysis")
        print(f"     ✓ No-model compressors work great for tone/morphology evaluation")
        print(f"     ✓ Try --mode lightweight if you get GPU access")
    elif args.mode in ('no_model',):
        print(f"     ✓ Compression runs 100% on CPU (0 VRAM)")
        print(f"     ✓ Generation uses INT4 model (~5GB VRAM)")
        print(f"     ✓ You still have ~{hw['vram_gb']-5:.0f}GB free")
    elif args.mode == 'lightweight':
        print(f"     ✓ Consider adding --tiny-scorer smollm2-135m for better quality")
        print(f"     ✓ Keep --max-new-tokens 128 to save KV cache VRAM")
    elif args.mode == 'full':
        print(f"     ✓ You have enough VRAM for full pipeline")
        print(f"     ✓ Add --ratios 2,4,6,8,10 for comprehensive evaluation")


if __name__ == '__main__':
    main()
