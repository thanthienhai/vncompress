#!/usr/bin/env python3
"""measure_token_inflation.py -- wave-2 E9: measure P3 (Vietnamese token inflation).

The wave-1 report left P3 ("the same content needs 1.5-2.0x more tokens in
Vietnamese than in English") as the one premise never actually measured. This
script measures it directly on VCC-Bench's cross-lingual pairs (which contain
parallel Vietnamese/English text), using the existing
linguistics.TokenInflationAnalyzer:

    TIR = tokens(vi_text) / tokens(en_text)

and reports the distribution across pairs. It also motivates E-series adaptive
compression rates (Vietnamese may warrant a different ratio than English).

Usage:
    python scripts/measure_token_inflation.py \
        --data-path data/benchmark/vcc_bench_v2.json \
        --tokenizer Qwen/Qwen2.5-7B-Instruct

    # Compare two tokenizers (e.g. a Vietnamese-native one vs a general one):
    python scripts/measure_token_inflation.py --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --en-tokenizer gpt2

No GPU needed -- tokenizers only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vncompress.linguistics import TokenInflationAnalyzer, is_vietnamese


def _split_lang(text_a: str, text_b: str):
    """Return (vi_text, en_text) from a (context, reference) pair by language
    detection, or (None, None) if they are not a VI/EN pair."""
    a_vi, b_vi = is_vietnamese(text_a), is_vietnamese(text_b)
    if a_vi and not b_vi:
        return text_a, text_b
    if b_vi and not a_vi:
        return text_b, text_a
    return None, None


def _extract_pairs(data: dict):
    """Yield (vi_text, en_text) pairs from cross-lingual VCC-Bench samples."""
    pairs = []
    for raw in data.get('samples', []):
        if raw.get('task') != 'cross_lingual':
            continue
        vi, en = _split_lang(raw.get('context', ''), raw.get('reference_answer', ''))
        if vi and en:
            pairs.append((vi, en))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Measure Vietnamese token inflation (VCC-Bench cross-lingual).')
    parser.add_argument('--data-path', default='data/benchmark/vcc_bench_v2.json',
                        help='VCC-Bench JSON (falls back to v1 if v2 is absent).')
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-7B-Instruct',
                        help='Tokenizer for the Vietnamese side (and English side unless --en-tokenizer is set).')
    parser.add_argument('--en-tokenizer', default=None,
                        help='Optional separate tokenizer for the English side (defaults to --tokenizer).')
    parser.add_argument('--out', default=None, help='Optional path to write a JSON report.')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        alt = data_path.replace('vcc_bench_v2.json', 'vcc_bench_v1.json')
        if os.path.exists(alt):
            print(f"[WARN] {data_path} not found; using {alt}")
            data_path = alt
        else:
            parser.error(f"Dataset not found: {data_path}")

    from transformers import AutoTokenizer

    vi_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    en_tok = AutoTokenizer.from_pretrained(args.en_tokenizer) if args.en_tokenizer else vi_tok

    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    pairs = _extract_pairs(data)
    if not pairs:
        print("[WARN] No cross-lingual VI/EN pairs found. TIR needs parallel text.")
        return

    analyzer = TokenInflationAnalyzer(vi_tokenizer=vi_tok, en_tokenizer=en_tok)
    vi_texts = [p[0] for p in pairs]
    en_texts = [p[1] for p in pairs]
    stats = analyzer.estimate_tir_batch(vi_texts, en_texts)
    per_sample = [analyzer.compute_tir(vi, en) for vi, en in pairs]

    print("=" * 60)
    print("Vietnamese Token Inflation Ratio (TIR = tokens_vi / tokens_en)")
    print("=" * 60)
    print(f"  pairs        : {len(pairs)}")
    print(f"  vi tokenizer : {args.tokenizer}")
    print(f"  en tokenizer : {args.en_tokenizer or args.tokenizer}")
    print(f"  mean TIR     : {stats['mean']:.3f}")
    print(f"  min / max    : {stats['min']:.3f} / {stats['max']:.3f}")
    print(f"  std          : {stats['std']:.3f}")
    print("-" * 60)
    print("Paper claim P3: 1.5-2.0x. Compare the mean above.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({
                'data_path': data_path, 'vi_tokenizer': args.tokenizer,
                'en_tokenizer': args.en_tokenizer or args.tokenizer,
                'n_pairs': len(pairs), 'stats': stats, 'per_sample_tir': per_sample,
            }, f, ensure_ascii=False, indent=2)
        print(f"Wrote report: {args.out}")


if __name__ == '__main__':
    main()
