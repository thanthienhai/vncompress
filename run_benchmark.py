#!/usr/bin/env python3
"""
VCC-Bench — Vietnamese Context Compression Benchmark
=====================================================
Main evaluation script for comparing compression methods on Vietnamese texts.

Usage:
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --methods tone_aware,combined
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --ratios 2,4,8
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --quick  # Quick demo

This script:
  1. Loads model + tokenizer
  2. Creates a demo Vietnamese dataset (or loads from file)
  3. Runs all compression methods at multiple ratios
  4. Prints comparison table and saves detailed results
"""

import argparse
import json
import os
import sys
import time
import torch

# Add parent to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress.evaluation import (
    VCCBench, VCCBenchConfig, VCCBenchSample,
    CompressionMetrics, evaluate_compression,
)
from vncompress.compressors import (
    create_compressor,
    COMPRESSOR_REGISTRY,
)
from vncompress.tone_aware import (
    VietnameseToneAnalyzer,
    is_vietnamese,
    get_tone_analyzer,
)


# ============================================================================
# Demo Dataset: Vietnamese long-context samples
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
            "ô nhiễm, sự cố, suy thoái môi trường. Bảo vệ môi trường gắn kết hài hòa "
            "với phát triển kinh tế, an sinh xã hội, bảo đảm quyền trẻ em, thúc đẩy "
            "bình đẳng giới và phát triển bền vững. Điều 5 quy định về chính sách "
            "của Nhà nước về bảo vệ môi trường bao gồm: tạo điều kiện thuận lợi cho "
            "tổ chức, hộ gia đình, cá nhân tham gia hoạt động bảo vệ môi trường; "
            "đẩy mạnh tuyên truyền, giáo dục, nâng cao nhận thức về bảo vệ môi trường."
        ) * 3,  # Repeat to create long context
        query="Nguyên tắc bảo vệ môi trường được quy định như thế nào trong Luật?",
        reference_answer=(
            "Nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường là quyền và "
            "trách nhiệm của mọi tổ chức, cá nhân; hoạt động bảo vệ môi trường phải "
            "thường xuyên, công khai, minh bạch; ưu tiên phòng ngừa ô nhiễm; gắn kết "
            "với phát triển kinh tế và an sinh xã hội."
        ),
        context_length=0,  # Will be computed
    ),
    VCCBenchSample(
        task='long_document_qa',
        context=(
            "Trí tuệ nhân tạo (AI) đang phát triển nhanh chóng và có tác động sâu rộng "
            "đến mọi mặt của đời sống xã hội. Các mô hình ngôn ngữ lớn như GPT, Gemini, "
            "và Claude đã đạt được những tiến bộ vượt bậc trong việc hiểu và sinh văn bản "
            "tiếng Việt. Tuy nhiên, việc xử lý các văn bản tiếng Việt dài vẫn còn nhiều "
            "thách thức do đặc điểm ngôn ngữ đơn lập, có thanh điệu và nhiều từ ghép. "
            "Các phương pháp nén ngữ cảnh hiện tại chủ yếu được phát triển cho tiếng Anh "
            "và chưa được kiểm chứng trên tiếng Việt. Nghiên cứu này đề xuất các phương "
            "pháp nén có nhận thức về thanh điệu và hình thái từ cho tiếng Việt. "
            "Kết quả thực nghiệm cho thấy các phương pháp đề xuất cải thiện đáng kể "
            "tỉ lệ bảo toàn thông tin so với các phương pháp truyền thống."
        ) * 3,
        query="Những thách thức chính khi xử lý văn bản tiếng Việt dài là gì?",
        reference_answer=(
            "Thách thức chính khi xử lý văn bản tiếng Việt dài bao gồm: đặc điểm "
            "ngôn ngữ đơn lập, có thanh điệu, và nhiều từ ghép. Các phương pháp "
            "nén ngữ cảnh hiện tại chưa được kiểm chứng trên tiếng Việt."
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
            "an toàn thực phẩm và giấy khám sức khỏe của chủ cơ sở. Ngoài ra cần "
            "đăng ký với cơ quan quản lý thực phẩm địa phương."
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
            "thông tin và viễn thông. Qua 15 năm phát triển, công ty đã mở rộng "
            "sang các lĩnh vực trí tuệ nhân tạo, dữ liệu lớn và điện toán đám mây. "
            "Năm 2025, công ty đạt doanh thu 5000 tỷ đồng và lợi nhuận sau thuế "
            "là 850 tỷ đồng. MẬT KHẨU BÍ MẬT: VIETCOMPRESS2026 "
            "Công ty hiện có hơn 5000 nhân viên làm việc tại 10 chi nhánh trên "
            "toàn quốc và 3 văn phòng đại diện tại nước ngoài."
        ) * 5,  # Long context with needle hidden
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
            "Agent: Dựa trên thông tin thời tiết, tôi khuyên bạn nên mang ô.\n"
            "User: Còn Đà Nẵng thì sao?\n"
            "Agent: Để tôi tra cứu.\n"
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
            "direct investment. Key sectors include manufacturing, technology, and "
            "agriculture. Nền kinh tế Việt Nam đã thể hiện sức phục hồi đáng kể. "
            "Tăng trưởng GDP đạt 7.2% năm 2025, được thúc đẩy bởi xuất khẩu mạnh "
            "và đầu tư trực tiếp nước ngoài. Các lĩnh vực chính bao gồm sản xuất, "
            "công nghệ và nông nghiệp."
        ) * 2,
        query="Tăng trưởng GDP của Việt Nam năm 2025 là bao nhiêu?",
        reference_answer="Tăng trưởng GDP của Việt Nam năm 2025 là 7.2%.",
        context_length=0,
    ),
]


# ============================================================================
# Main benchmark
# ============================================================================

def setup_model_and_tokenizer(model_name: str, device: str = 'cuda'):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
    }
    
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    print(f"Model loaded. Vocab size: {len(tokenizer)}, "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    
    return model, tokenizer


def run_benchmark(
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    device: str = 'cuda',
    methods: list = None,
    ratios: list = None,
    output_dir: str = './results',
    quick: bool = False,
):
    """Run the full VCC-Bench evaluation."""
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(model_name, device)
    
    # Configure benchmark
    config = VCCBenchConfig(
        methods=methods or ['none', 'random', 'llmlingua', 'tone_aware', 'morphology_aware', 'combined'],
        compression_ratios=ratios or ([2.0] if quick else [2.0, 4.0, 8.0]),
        output_dir=output_dir,
        device=device,
        max_new_tokens=128 if quick else 256,
    )
    
    bench = VCCBench(config)
    
    # Add samples
    samples = VIETNAMESE_DEMO_SAMPLES[:2] if quick else VIETNAMESE_DEMO_SAMPLES
    
    # Compute context lengths
    for sample in samples:
        sample.context_length = len(tokenizer.encode(sample.context))
    
    bench.add_samples(samples)
    
    print(f"\nVCC-Bench Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Methods: {config.methods}")
    print(f"  Ratios: {[f'{r}x' for r in config.compression_ratios]}")
    print(f"  Samples: {bench.total_samples}")
    print(f"  Tasks: {list(bench.samples.keys())}")
    
    # Tone analysis summary
    tone_analyzer = get_tone_analyzer()
    vi_samples = [s for s in samples if is_vietnamese(s.context[:500])]
    if vi_samples:
        sample_tokens = []
        for s in vi_samples[:1]:
            ids = tokenizer.encode(s.context)
            for tid in ids[:100]:
                t = tokenizer.decode([tid])
                sample_tokens.append(t.strip())
        
        tone_stats = tone_analyzer.analyze_tokens(sample_tokens)
        avg_weight = sum(t.preservation_weight for t in tone_stats) / max(len(tone_stats), 1)
        print(f"\n  Tone Analysis (first 100 tokens):")
        print(f"    Avg tone preservation weight: {avg_weight:.3f}")
        print(f"    Tone-bearing tokens: {sum(1 for t in tone_stats if t.tones_present)}/{len(tone_stats)}")
    
    # Run evaluation
    def make_compressor(method_name: str):
        return create_compressor(
            method_name, tokenizer, model,
            config=None,  # Will use default
            device=device,
        )
    
    results = bench.evaluate(
        compressor_fn=make_compressor,
        model=model,
        tokenizer=tokenizer,
    )
    
    # Print summary
    bench.print_summary(results)
    
    # Save full results
    results_path = os.path.join(output_dir, "vcc_bench_results.json")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable
    serializable = {}
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            serializable[method] = method_results
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {results_path}")
    
    # Generate report
    report = bench.generate_report(results)
    report_path = os.path.join(output_dir, "vcc_bench_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    return results


def quick_demo(model_name: str = 'Qwen/Qwen2.5-7B-Instruct', device: str = 'cuda'):
    """Quick demo with a single sample to verify the pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*60)
    print("VNCOMPRESS — Quick Demo")
    print("="*60)
    
    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto' if device == 'cuda' else None,
    )
    model.eval()
    
    # Use first demo sample
    sample = VIETNAMESE_DEMO_SAMPLES[0]
    
    print(f"\nOriginal context: {len(tokenizer.encode(sample.context))} tokens")
    print(f"Query: {sample.query}")
    print(f"Reference: {sample.reference_answer}\n")
    
    # Test each method
    methods_to_test = ['none', 'llmlingua', 'tone_aware', 'morphology_aware', 'combined']
    
    for method_name in methods_to_test:
        print(f"\n--- {method_name} ---")
        
        compressor = create_compressor(
            method_name, tokenizer, model, device=device
        )
        
        metric = evaluate_compression(
            compressor, model, tokenizer,
            input_text=sample.context,
            query=sample.query,
            reference=sample.reference_answer,
            ratio=4.0,
        )
        
        print(f"  Compression: {metric.compression_ratio:.1f}x")
        print(f"  Token savings: {metric.token_savings_pct:.1f}%")
        print(f"  Processing: {metric.processing_time_ms:.1f}ms")
        print(f"  ROUGE-L F1: {metric.rouge_l_f1:.4f}" if metric.rouge_l_f1 is not None else "  ROUGE-L: N/A")
        print(f"  BLEU: {metric.bleu_score:.4f}" if metric.bleu_score is not None else "  BLEU: N/A")
        print(f"  Quality: {metric.quality_score:.4f}")
        print(f"  Efficiency: {metric.efficiency_score:.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='VCC-Bench: Vietnamese Context Compression Benchmark'
    )
    
    parser.add_argument(
        '--model', type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name (HuggingFace)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run on'
    )
    parser.add_argument(
        '--methods', type=str, default=None,
        help='Comma-separated method names (default: all)'
    )
    parser.add_argument(
        '--ratios', type=str, default=None,
        help='Comma-separated compression ratios (default: 2,4,8)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick evaluation with fewer ratios and samples'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run a quick demo with detailed output per method'
    )
    parser.add_argument(
        '--list-methods', action='store_true',
        help='List available compression methods'
    )
    
    args = parser.parse_args()
    
    if args.list_methods:
        print("Available compression methods:")
        for name, cls in COMPRESSOR_REGISTRY.items():
            print(f"  {name:<25} -> {cls.__name__}")
        return
    
    if args.demo:
        quick_demo(args.model, args.device)
        return
    
    methods = args.methods.split(',') if args.methods else None
    ratios = None
    if args.ratios:
        try:
            ratios = [float(r) for r in args.ratios.split(',')]
        except ValueError:
            parser.error(f"Invalid compression ratios: '{args.ratios}'. "
                         "Use comma-separated numbers, e.g. --ratios 2,4,8")
    
    run_benchmark(
        model_name=args.model,
        device=args.device,
        methods=methods,
        ratios=ratios,
        output_dir=args.output_dir,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()
