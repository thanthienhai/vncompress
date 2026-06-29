#!/usr/bin/env python3
"""
Test suite for GTX 1060 6GB / CPU-only: pure-Python modules.
Uses only ASCII for output — avoids Windows Unicode terminal issues.
"""
import sys, os, time, io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PASS = 0
FAIL = 0

def check(desc, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {desc}")
        PASS += 1
    else:
        print(f"  [FAIL] {desc}")
        FAIL += 1

print("=" * 60)
print("  VNCOMPRESS -- Pure Python Test Suite")
print("=" * 60)

# --- Imports ---
print("\n--- Imports ---")
try:
    from vncompress.tone_aware import (
        VietnameseToneAnalyzer, TokenToneInfo, ToneInfo,
        is_vietnamese, strip_tone, extract_tone_marks, get_tone_analyzer,
    )
    check("[tone_aware core]", True)
except Exception as e:
    check(f"[tone_aware core] {e}", False)

try:
    from vncompress.morphology import (
        MorphologyAnalyzer, WordClass, WordInfo,
        MorphologyConfig, get_morphology_analyzer,
    )
    check("[morphology]", True)
except Exception as e:
    check(f"[morphology] {e}", False)

try:
    from vncompress.compressors.no_model import (
        NoModelToneCompressor, NoModelMorphCompressor,
        NoModelCombinedCompressor, NoModelBaselineCompressor,
        evaluate_no_model, NoModelResult,
    )
    check("[no_model]", True)
except Exception as e:
    check(f"[no_model] {e}", False)

# Lazy __getattr__ should correctly block torch-dependent imports
try:
    from vncompress.tone_aware import ToneAwareConfig
    check("[lazy: ToneAwareConfig loaded (torch installed)]", True)
except ImportError:
    check("[lazy: ToneAwareConfig blocked (no torch, correct)]", True)

# --- Tone Analyzer ---
print("\n--- VietnameseToneAnalyzer ---")
analyzer = get_tone_analyzer(alpha=0.5, beta=0.3, gamma=0.4)

# Vietnamese accented tokens
tokens = ['\u0111\u00e3', 'h\u1ecdc', 'sinh', 'xinh_x\u1eafn', 'c\u1ee7a', 'v\u00e0']
infos = analyzer.analyze_tokens(tokens, window_size=2)
check(f"analyze: {len(infos)} results", len(infos) == len(tokens))
check("token da: has dominant_tone", infos[0].dominant_tone is not None)
check("token da: weight > 1.0", infos[0].preservation_weight > 1.0)
check("token sinh: tone = ngang", infos[2].dominant_tone == 'ngang')
check("token sinh: density = 0", infos[2].tone_density == 0.0)

# Edge cases
check("empty list -> []", len(analyzer.analyze_tokens([])) == 0)
check("empty str tone = ngang", analyzer.get_dominant_tone('') == 'ngang')
check("empty str density = 0", analyzer.compute_tone_density('') == 0.0)
check("empty str weight = 1.0", abs(analyzer.compute_preservation_weight('') - 1.0) < 0.001)

# Singleton: new kwargs -> new instance
a2 = get_tone_analyzer(alpha=0.9)
check("singleton respects kwargs", abs(a2.alpha - 0.9) < 0.01)

# Utilities
vi_text = '\u0110\u00e2y l\u00e0 ti\u1ebfng Vi\u1ec7t c\u00f3 d\u1ea5u'
check("is_vietnamese(diacritics) = True", is_vietnamese(vi_text) is True)
check("is_vietnamese(Hello) = False", is_vietnamese('Hello world') is False)
check("is_vietnamese('') = False", is_vietnamese('') is False)
check("strip_tone returns string", isinstance(strip_tone('H\u00f4m nay'), str))
check("extract_tone_marks returns list", isinstance(extract_tone_marks('H\u00f4m'), list))

# Stress: 2500 tokens should be fast
long_tokens = ["xin"] * 2500
t0 = time.time()
result = analyzer.analyze_tokens(long_tokens, window_size=1)
elapsed = time.time() - t0
check(f"2500 tokens in {elapsed*1000:.0f}ms (< 500ms)", elapsed < 0.5)

# --- Morphology Analyzer ---
print("\n--- MorphologyAnalyzer ---")
morph = get_morphology_analyzer()
tokens = ['\u0111\u00e3', 'h\u1ecdc', 'sinh', 'xinh_x\u1eafn',
          'c\u1ee7a', 'm\u00e1y_t\u00ednh', 'xyz', 'v\u00e0']
infos = morph.classify_batch(tokens)
check(f"classify: {len(infos)} results", len(infos) == len(tokens))
check("da -> FUNC", infos[0].word_class == WordClass.FUNC)
check("hoc -> CONTENT", infos[1].word_class == WordClass.CONTENT)
check("xinh_xan -> REDUP", infos[3].word_class == WordClass.REDUP)
check("cua -> FUNC", infos[4].word_class == WordClass.FUNC)
check("may_tinh -> COMPOUND", infos[5].word_class == WordClass.COMPOUND)
check("va -> FUNC", infos[7].word_class == WordClass.FUNC)

# Config
cfg = MorphologyConfig(f_func=0.4, f_content=1.2, f_compound=1.5)
check("FUNC multiplier", abs(morph.get_preservation_multiplier(infos[0], cfg) - 0.4) < 0.01)
check("COMPOUND multiplier", abs(morph.get_preservation_multiplier(infos[5], cfg) - 1.5) < 0.01)
check("empty classify -> []", len(morph.classify_batch([])) == 0)

# Duplicates in function words
from vncompress.morphology.merge_policy import VIETNAMESE_FUNCTION_WORDS
dupes = [w for w in VIETNAMESE_FUNCTION_WORDS
         if list(VIETNAMESE_FUNCTION_WORDS).count(w) > 1]
check(f"no duplicate function words ({len(dupes)} found)", len(dupes) == 0)

# --- No-Model Compressors ---
print("\n--- No-Model Compressors ---")
class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text]
    def decode(self, ids):
        if isinstance(ids, list):
            return ''.join(chr(min(id % 1000, 127)) for id in ids)
        return chr(ids % 1000 if isinstance(ids, list) else min(ids % 1000, 127))

tk = FakeTokenizer()
text = "Hom nay troi dep, chung toi di dao trong cong vien"
ids = tk.encode(text)
target = 2.0

for mode in ['first', 'random', 'word_length', 'every_nth']:
    comp = NoModelBaselineCompressor(tk, mode=mode)
    r = comp.compress(ids, target_ratio=target)
    check(f"baseline_{mode}: compresses", r.compressed_length < r.original_length)

comp = NoModelToneCompressor(tk)
r = comp.compress(ids, target_ratio=4.0)
check("tone_only: ratio ~4", 3.0 < r.compression_ratio < 5.5)

comp = NoModelMorphCompressor(tk)
r = comp.compress(ids, target_ratio=4.0)
check("morph_only: saves tokens", r.token_savings_pct > 0)

comp = NoModelCombinedCompressor(tk)
r = comp.compress(ids, target_ratio=4.0, tone_weight=0.5)
check("combined: saves tokens", r.token_savings_pct > 0)

# evaluate_no_model
old_stdout = sys.stdout
sys.stdout = io.StringIO()
results = evaluate_no_model(text, tk, target_ratio=4.0)
sys.stdout = old_stdout
check(f"evaluate_no_model: {len(results)} methods", len(results) == 6)
for name, r in results.items():
    check(f"{name}: has result", r.compression_ratio > 0)

# --- Vietnamese Linguistics ---
print("\n--- Vietnamese Linguistics ---")
try:
    from vncompress.tone_aware.vietnamese_linguistics import (
        get_syllable_tone, VietnameseWordSegmenter,
        is_vietnamese_function_word_extended, normalize_vietnamese_word,
        is_critical_pattern,
    )
    check("[linguistics import]", True)

    check("get_syllable_tone: dep=nang",
          get_syllable_tone('\u0111\u1eb9p') == 'n\u1eb7ng')
    check("get_syllable_tone: khong=ngang",
          get_syllable_tone('kh\u00f4ng') == 'ngang')
    check("get_syllable_tone: xyz=ngang",
          get_syllable_tone('xyz') == 'ngang')

    check("is_func_word_extended: da",
          is_vietnamese_function_word_extended('\u0111\u00e3'))
    check("is_func_word_extended: ko",
          is_vietnamese_function_word_extended('ko'))
    check("normalize: ko->khong",
          normalize_vietnamese_word('ko') == 'kh\u00f4ng')

    check("is_critical: Dieu 4",
          is_critical_pattern('\u0110i\u1ec1u 4'))
    check("is_critical: email",
          is_critical_pattern('test@email.com'))
    check("not is_critical: normal",
          not is_critical_pattern('normal'))

    seg = VietnameseWordSegmenter(use_external=False)
    groups = seg.group_subword_tokens(['h\u1ee3p', 't\u00e1c', 'x\u00e3'])
    check("word segmenter: finds hop_tac_xa",
          'h\u1ee3p_t\u00e1c_x\u00e3' in groups)
except Exception as e:
    check(f"[linguistics] {e}", False)

# --- Evaluation Metrics ---
print("\n--- Evaluation Metrics ---")
try:
    from vncompress.evaluation.metrics import compute_exact_match, compute_rouge_l
    check("[metrics import]", True)
    check("exact_match identical", compute_exact_match(["a"], ["a"]) == 1.0)
    check("exact_match different", compute_exact_match(["a"], ["b"]) == 0.0)
    r = compute_rouge_l(["xin chao cac ban"], ["xin chao cac ban"])
    check("rouge_l identical: f1>0.9", r['rougeL_f1'] > 0.9)
    r = compute_rouge_l(["xin chao"], ["tam biet"])
    check("rouge_l different: f1<0.5", r['rougeL_f1'] < 0.5)
except Exception as e:
    check(f"[metrics] {e}", False)

# --- Summary ---
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed  ({PASS+FAIL} total)")
print("=" * 60)
if FAIL:
    print(f"\n  !! {FAIL} FAILURES !!")
    sys.exit(1)
else:
    print("\n  All clear - no errors!")
    sys.exit(0)
