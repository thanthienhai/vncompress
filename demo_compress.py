#!/usr/bin/env python3
"""
Demo: Hiển thị kết quả nén văn bản tiếng Việt với các phương pháp khác nhau.
"""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from transformers import AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelMorphCompressor,
    NoModelCombinedCompressor, NoModelBaselineCompressor,
)
from vncompress.tone_aware import VietnameseToneAnalyzer, get_tone_analyzer
from vncompress.morphology import MorphologyAnalyzer, get_morphology_analyzer

# ── Vietnamese sample text ──
TEXT = (
    "Lu\u1eadt B\u1ea3o v\u1ec7 M\u00f4i tr\u01b0\u1eddng n\u0103m 2020 quy \u0111\u1ecbnh "
    "v\u1ec1 ho\u1ea1t \u0111\u1ed9ng b\u1ea3o v\u1ec7 m\u00f4i tr\u01b0\u1eddng, "
    "quy\u1ec1n, ngh\u0129a v\u1ee5 v\u00e0 tr\u00e1ch nhi\u1ec7m c\u1ee7a c\u01a1 quan, "
    "t\u1ed5 ch\u1ee9c, c\u1ed9ng \u0111\u1ed3ng d\u00e2n c\u01b0, "
    "h\u1ed9 gia \u0111\u00ecnh v\u00e0 c\u00e1 nh\u00e2n trong ho\u1ea1t \u0111\u1ed9ng "
    "b\u1ea3o v\u1ec7 m\u00f4i tr\u01b0\u1eddng. "
    "\u0110i\u1ec1u 4 quy \u0111\u1ecbnh nguy\u00ean t\u1eafc b\u1ea3o v\u1ec7 m\u00f4i "
    "tr\u01b0\u1eddng bao g\u1ed3m: b\u1ea3o v\u1ec7 m\u00f4i tr\u01b0\u1eddng l\u00e0 "
    "quy\u1ec1n, ngh\u0129a v\u1ee5 v\u00e0 tr\u00e1ch nhi\u1ec7m c\u1ee7a m\u1ecdi c\u01a1 quan, "
    "t\u1ed5 ch\u1ee9c, c\u1ed9ng \u0111\u1ed3ng d\u00e2n c\u01b0, h\u1ed9 gia \u0111\u00ecnh "
    "v\u00e0 c\u00e1 nh\u00e2n. Ho\u1ea1t \u0111\u1ed9ng b\u1ea3o v\u1ec7 m\u00f4i tr\u01b0\u1eddng "
    "ph\u1ea3i \u0111\u01b0\u1ee3c ti\u1ebfn h\u00e0nh th\u01b0\u1eddng xuy\u00ean, "
    "c\u00f4ng khai, minh b\u1ea1ch."
) * 2

input_ids = tokenizer.encode(TEXT, add_special_tokens=False)
tokens = []
for tid in input_ids:
    t = tokenizer.decode([tid]).replace('\u2581', ' ').replace('Ġ', ' ').strip()
    tokens.append(t)

print("=" * 70)
print("  VNCOMPRESS - Compression Demo")
print("=" * 70)
print(f"\n  Original: {len(input_ids)} tokens")
print(f"  Text preview: {TEXT[:120]}...")
print()

# ── Tone & Morphology analysis ──
tone_analyzer = get_tone_analyzer()
morph_analyzer = get_morphology_analyzer()

tone_infos = tone_analyzer.analyze_tokens(tokens)
morph_infos = morph_analyzer.classify_batch(tokens)

# Show tone/morphology for first 15 tokens
print("  ┌─ Token Classification (first 15) ─────────────────────────┐")
print(f"  │ {'Token':<12} {'Tone':>8} {'W_tone':>7} {'Class':<12} │")
print("  ├───────────────────────────────────────────────────────────┤")
for ti, mi in zip(tone_infos[:15], morph_infos[:15]):
    tone_name = ti.dominant_tone or 'ngang'
    w = ti.preservation_weight
    cls = mi.word_class.value
    token_display = ti.token if len(ti.token) <= 10 else ti.token[:9] + '…'
    print(f"  │ {token_display:<12} {tone_name:>8} {w:>7.3f} {cls:<12} │")
print("  └───────────────────────────────────────────────────────────┘")

# ── Run all compressors ──
RATIO = 4.0
compressors = [
    ('Không nén (baseline)',     NoModelBaselineCompressor(tokenizer, mode='first')),
    ('Ngẫu nhiên (random)',      NoModelBaselineCompressor(tokenizer, mode='random')),
    ('Tone-only',                NoModelToneCompressor(tokenizer)),
    ('Morphology-only',          NoModelMorphCompressor(tokenizer)),
    ('Tone + Morph (combined)',  NoModelCombinedCompressor(tokenizer)),
]

print(f"\n  {'='*70}")
print(f"  COMPRESSION RESULTS (target ratio: {RATIO:.0f}x)")
print(f"  {'='*70}")

for method_name, comp in compressors:
    result = comp.compress(input_ids, target_ratio=RATIO)

    # Decode compressed
    compressed_tokens = []
    for tid in result.compressed_ids:
        t = tokenizer.decode([tid]).replace('\u2581', ' ').replace('Ġ', ' ').strip()
        compressed_tokens.append(t)

    # Count tone/morph stats on compressed
    compressed_tone_infos = tone_analyzer.analyze_tokens(compressed_tokens)
    compressed_morph_infos = morph_analyzer.classify_batch(compressed_tokens)

    n_orig_func = sum(1 for mi in morph_infos if mi.word_class.value == 'function')
    n_orig_content = sum(1 for mi in morph_infos if mi.word_class.value == 'content')
    n_comp_func = sum(1 for mi in compressed_morph_infos if mi.word_class.value == 'function')
    n_comp_content = sum(1 for mi in compressed_morph_infos if mi.word_class.value == 'content')

    n_orig_tone = sum(1 for ti in tone_infos if ti.tones_present)
    n_comp_tone = sum(1 for ti in compressed_tone_infos if ti.tones_present)

    print(f"\n  ┌─ {method_name} ──────────────────────────────────────────────┐")
    print(f"  │ Tokens:    {result.original_length:>4d} → {result.compressed_length:<4d}   "
          f"({result.compression_ratio:.1f}x, {result.token_savings_pct:.0f}% saved)")
    print(f"  │ Thời gian: {result.processing_time_ms:.1f}ms")
    print(f"  │ Content words kept:   {n_comp_content}/{n_orig_content} "
          f"({n_comp_content/max(n_orig_content,1)*100:.0f}%)")
    print(f"  │ Function words kept:  {n_comp_func}/{n_orig_func} "
          f"({n_comp_func/max(n_orig_func,1)*100:.0f}%)")
    print(f"  │ Tone tokens kept:     {n_comp_tone}/{n_orig_tone} "
          f"({n_comp_tone/max(n_orig_tone,1)*100:.0f}%)")

    # Show compressed text
    compressed_text = tokenizer.decode(result.compressed_ids, skip_special_tokens=True)
    # Show first 300 chars
    if len(compressed_text) > 300:
        preview = compressed_text[:300] + "..."
    else:
        preview = compressed_text
    print(f"  ├───────────────────────────────────────────────────────────┤")
    print(f"  │ Kết quả nén:")
    # Word-wrap the preview
    for i in range(0, len(preview), 55):
        print(f"  │ {preview[i:i+55]}")
    print(f"  └───────────────────────────────────────────────────────────┘")

# ── Summary table ──
print(f"\n  {'='*70}")
print(f"  {'BẢNG SO SÁNH':^68}")
print(f"  {'='*70}")
print(f"  {'Phương pháp':<22} {'Tokens':>8} {'CR':>5} {'Save':>6} {'Content':>8} {'Tone':>6}")
print(f"  {'-'*60}")

for method_name, comp in compressors:
    result = comp.compress(input_ids, target_ratio=RATIO)
    ct = []
    for tid in result.compressed_ids:
        ct.append(tokenizer.decode([tid]).replace('\u2581', ' ').replace('Ġ', ' ').strip())
    cmi = morph_analyzer.classify_batch(ct)
    cti = tone_analyzer.analyze_tokens(ct)

    n_cc = sum(1 for mi in cmi if mi.word_class.value == 'content')
    n_ct = sum(1 for ti in cti if ti.tones_present)
    content_pct = n_cc/max(n_orig_content,1)*100
    tone_pct = n_ct/max(n_orig_tone,1)*100

    print(f"  {method_name:<22} {result.original_length:>3}->{result.compressed_length:<3} "
          f"{result.compression_ratio:>4.1f}x {result.token_savings_pct:>5.0f}% "
          f"{content_pct:>7.0f}% {tone_pct:>5.0f}%")
print(f"  {'='*70}")
