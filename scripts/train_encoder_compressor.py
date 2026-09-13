#!/usr/bin/env python3
"""train_encoder_compressor.py -- wave-2 E6/E11: distill a keep/drop token
classifier for EncoderClassifierCompressor (LLMLingua-2 style).

Recipe (offline, once): for each training text, score tokens with a strong
*teacher* causal LM's windowed perplexity (the same signal LLMLingua uses), keep
the top `1/ratio` fraction, and treat "kept" as the positive class. Those
teacher keep/drop decisions are projected onto a Vietnamese *encoder*
(PhoBERT / XLM-R) token classifier via character spans, and the encoder is
fine-tuned to reproduce them.

Why (wave-1 report §8.6, §5): the 4B generative scorer loses to a 0.5B one and
costs ~8 GB just to rank tokens. A distilled bidirectional encoder does the same
job in one forward pass at a fraction of the cost -- the "win on cost" arm. Once
trained, point EncoderClassifierCompressor at --output-dir via encoder_path
(or `benchmark.py --methods encoder` after wiring the checkpoint).

Usage:
    python scripts/train_encoder_compressor.py \
        --train-data-path data/benchmark/training_corpus_v1.json \
        --encoder-id vinai/phobert-base \
        --teacher-model Qwen/Qwen2.5-0.5B-Instruct \
        --ratio 4 --epochs 2 --output-dir models/encoder_compressor

Needs a GPU for the teacher pass in practice; the encoder fine-tune is light.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_labels_for_text(text, teacher_model, teacher_tok, enc_tok, ratio, max_len):
    """Return (encoder_input_ids, encoder_labels) for one text.

    Labels: 1 (keep) / 0 (drop) / -100 (ignore: special or uncovered tokens),
    derived from the teacher's top-`1/ratio` perplexity tokens.
    """
    import torch

    from vncompress.compression import _token_spans, sliding_window_perplexity

    teacher_ids = teacher_tok.encode(text, add_special_tokens=False)
    if len(teacher_ids) < 4:
        return None
    scores = sliding_window_perplexity(teacher_model, teacher_ids)
    n = len(teacher_ids)
    keep_k = max(1, int(n / ratio))
    keep_idx = set(torch.topk(scores, min(keep_k, n)).indices.tolist())

    # Teacher token spans over the reconstructed text -> per-char keep mask.
    ttext, tspans, _ = _token_spans(teacher_tok, teacher_ids)
    keep_mask = torch.zeros(len(ttext))
    for i, (s, e) in enumerate(tspans):
        if i in keep_idx and e > s:
            keep_mask[s:e] = 1.0

    enc = enc_tok(ttext, return_offsets_mapping=True, add_special_tokens=True,
                  truncation=True, max_length=max_len)
    labels = []
    for (s, e) in enc['offset_mapping']:
        if e <= s:  # special token / empty span
            labels.append(-100)
        else:
            labels.append(1 if keep_mask[s:min(e, len(keep_mask))].mean() >= 0.5 else 0)
    return enc['input_ids'], labels


def main():
    ap = argparse.ArgumentParser(description='Distill a keep/drop encoder token classifier (wave-2 E6).')
    ap.add_argument('--train-data-path', default=None, help='JSON corpus (see vncompress.training.load_training_texts).')
    ap.add_argument('--encoder-id', default='vinai/phobert-base', help='Encoder to fine-tune (PhoBERT / XLM-R).')
    ap.add_argument('--teacher-model', default='Qwen/Qwen2.5-0.5B-Instruct', help='Causal LM whose perplexity keep-sets are distilled.')
    ap.add_argument('--ratio', type=float, default=4.0, help='Compression ratio used to derive the keep fraction (1/ratio).')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--max-texts', type=int, default=-1, help='Cap number of training texts (-1 = all).')
    ap.add_argument('--output-dir', default='models/encoder_compressor')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from vncompress.models import load_model
    from vncompress.training import load_training_texts

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f"Teacher: {args.teacher_model} | Encoder: {args.encoder_id} | ratio={args.ratio} | device={device}")

    teacher_model, teacher_tok = load_model(args.teacher_model, device=device,
                                            dtype='float16' if device == 'cuda' else 'float32')
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_id, use_fast=True)

    texts = load_training_texts(args.train_data_path)
    if args.max_texts > 0:
        texts = texts[:args.max_texts]
    print(f"Building distillation labels for {len(texts)} texts...")

    examples = []
    for text in texts:
        built = build_labels_for_text(text, teacher_model, teacher_tok, enc_tok, args.ratio, args.max_length)
        if built:
            examples.append(built)
    if not examples:
        raise RuntimeError("No usable training examples were built.")
    print(f"  {len(examples)} labeled examples")

    pad_id = enc_tok.pad_token_id or 0

    def collate(batch):
        width = min(max(len(ids) for ids, _ in batch), args.max_length)
        bsz = len(batch)
        input_ids = torch.full((bsz, width), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, width), dtype=torch.long)
        labels = torch.full((bsz, width), -100, dtype=torch.long)
        for r, (ids, labs) in enumerate(batch):
            L = min(len(ids), width)
            input_ids[r, :L] = torch.tensor(ids[:L])
            attn[r, :L] = 1
            labels[r, :L] = torch.tensor(labs[:L])
        return {'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = AutoModelForTokenClassification.from_pretrained(
        args.encoder_id, num_labels=2, id2label={0: 'drop', 1: 'keep'}, label2id={'drop': 0, 'keep': 1},
    )
    if device == 'cuda':
        model = model.to('cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  epoch {epoch + 1} step {step} loss {out.loss.item():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    enc_tok.save_pretrained(args.output_dir)
    print(f"Saved distilled encoder classifier to: {args.output_dir}")
    print(f"Use it: EncoderClassifierCompressor(tokenizer, encoder_path='{args.output_dir}')")


if __name__ == '__main__':
    main()
