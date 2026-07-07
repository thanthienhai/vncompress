#!/usr/bin/env python3
"""Demo: show what compressed Vietnamese text looks like with different methods."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vncompress.compressors.no_model import (
    NoModelToneCompressor, NoModelMorphCompressor,
    NoModelCombinedCompressor, NoModelBaselineCompressor,
)
from vncompress.tone_aware.vietnamese_tones import VietnameseToneAnalyzer, TONE_NAME_TO_ID
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tone_analyzer = VietnameseToneAnalyzer()

# Sample Vietnamese text
text = (
    "Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
    "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
    "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. "
    "Bảo vệ môi trường là quyền, nghĩa vụ và trách nhiệm của mọi cơ quan, "
    "tổ chức, cộng đồng dân cư, hộ gia đình và cá nhân."
)

input_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"\nOriginal: {len(input_ids)} tokens")
print(f"Text: {text[:200]}...")

def show_compression(name, compressor, ratio):
    result = compressor.compress(input_ids, target_ratio=ratio)
    comp_text = tokenizer.decode(result.compressed_ids, skip_special_tokens=False)
    comp_ids = result.compressed_ids

    # Compute TPR
    def get_tones(ids):
        tones = []
        for tid in ids:
            ts = tokenizer.decode([tid])
            ts = ts.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tone_name = tone_analyzer.get_dominant_tone(ts[:20])
            tones.append(TONE_NAME_TO_ID.get(tone_name or 'ngang', 0))
        return tones
    orig_tones = get_tones(input_ids)
    comp_tones = get_tones(comp_ids)
    orig_tone_count = sum(1 for t in orig_tones if t > 0)
    comp_tone_count = sum(1 for t in comp_tones if t > 0)
    tpr = comp_tone_count / max(orig_tone_count, 1)

    cr = len(input_ids) / max(len(comp_ids), 1)
    print(f"\n{'='*60}")
    print(f"{name} @ {ratio}x: {len(input_ids)} -> {len(comp_ids)} tokens (CR={cr:.1f}x, TPR={tpr:.3f})")
    print(f"{'='*60}")
    # Show decoded text (first 300 chars)
    print(comp_text[:400])
    if len(comp_text) > 400:
        print("...")

compressors = {
    "1. NO COMPRESSION (giữ N token đầu)": NoModelBaselineCompressor(tokenizer, mode="first"),
    "2. RANDOM (chọn ngẫu nhiên)": NoModelBaselineCompressor(tokenizer, mode="random"),
    "3. TONE-AWARE (ưu tiên token có dấu)": NoModelToneCompressor(tokenizer),
    "4. MORPH-AWARE (nén mạnh hư từ)": NoModelMorphCompressor(tokenizer),
    "5. COMBINED (tone + morphology)": NoModelCombinedCompressor(tokenizer),
}

for name, comp in compressors.items():
    show_compression(name, comp, ratio=4.0)

# Also show original text with tone marks highlighted
print(f"\n{'='*60}")
print("PHÂN TÍCH TONE CỦA TỪNG TOKEN (gốc)")
print(f"{'='*60}")
tokens = []
for tid in input_ids:
    ts = tokenizer.decode([tid])
    ts = ts.replace('\u2581', ' ').replace('Ġ', ' ').strip()
    tone_name = tone_analyzer.get_dominant_tone(ts[:20])
    tone_id = TONE_NAME_TO_ID.get(tone_name or 'ngang', 0)
    marker = ""
    if tone_id == 0:
        marker = "·"  # ngang = neutral
    elif tone_id == 1:
        marker = "`"  # huyền
    elif tone_id == 2:
        marker = "'"  # sắc
    elif tone_id == 3:
        marker = "?"  # hỏi
    elif tone_id == 4:
        marker = "~"  # ngã
    elif tone_id == 5:
        marker = "."  # nặng
    tokens.append(f"[{marker}]{ts}")

print(" ".join(tokens[:30]))
print(f"\nLegend: ·=ngang `=huyền '=sắc ?=hỏi ~=ngã .=nặng")

tone_counts = {}
for t in orig_tones:
    tone_name = {0:'ngang',1:'huyền',2:'sắc',3:'hỏi',4:'ngã',5:'nặng'}[t]
    tone_counts[tone_name] = tone_counts.get(tone_name, 0) + 1
print(f"\nTone distribution: {tone_counts}")
print(f"Total tokens with tone marks (non-ngang): {orig_tone_count}/{len(input_ids)}")
