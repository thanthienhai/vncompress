#!/usr/bin/env python
"""CPU-only smoke test for CI.

Imports the vncompress package and runs one compressor end-to-end against
a real HuggingFace tokenizer (no model weights, no GPU) -- catches
integration issues the mocked unit test suite (tests/) can't see, without
requiring any of the large models used by the full benchmark/training
scripts.
"""
import sys

from transformers import AutoTokenizer

from vncompress.compressors.no_model import NoModelCombinedCompressor


def main() -> int:
    print("Loading tokenizer (gpt2, tokenizer only -- no model weights)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text = (
        "Hôm nay chúng tôi đã đi học ở một ngôi trường đại học rất xa "
        "nhưng vẫn đến đúng giờ vì luôn luôn chuẩn bị kỹ càng."
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Encoded {len(input_ids)} tokens")

    compressor = NoModelCombinedCompressor(tokenizer)
    result = compressor.compress(input_ids, target_ratio=4.0)

    assert result.compressed_length > 0, "compression produced no tokens"
    assert result.compressed_length <= result.original_length, "compression grew the sequence"

    print(
        f"OK: {result.original_length} -> {result.compressed_length} tokens "
        f"({result.compression_ratio:.2f}x, {result.token_savings_pct:.0f}% saved) "
        f"in {result.processing_time_ms:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
