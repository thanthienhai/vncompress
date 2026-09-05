"""Shared fixtures for the vncompress test suite.

Tests use MockTokenizer instead of a real HuggingFace tokenizer so the
suite runs on CPU, offline, and in milliseconds -- no model download,
no GPU, no VRAM. Compressors only ever call `.encode`/`.decode` on the
tokenizer they're given, so a deterministic word-level stand-in is a
faithful substitute for these unit tests.
"""
import pathlib
import re
import shutil
import tempfile

import pytest

try:
    import torch

    # Pin to single-threaded CPU intra-op parallelism for the whole test
    # session, for deterministic CPU test timing in CI.
    torch.set_num_threads(1)
except ImportError:
    pass

try:
    from vncompress import linguistics as _linguistics

    # get_morphology_analyzer() is a lazily-created module-level singleton
    # (vncompress/linguistics.py). Its default use_word_segmentation=True
    # path loads underthesea's word segmenter, which is slow/network-dependent
    # on first use and was observed to hang or crash the process
    # intermittently in this environment. Pre-seeding the singleton here --
    # before any test triggers its lazy creation -- keeps the whole suite on
    # the fast, deterministic, dictionary-only classification path. Tests
    # that specifically need a differently-configured analyzer construct
    # MorphologyAnalyzer directly.
    _linguistics._default_morph_analyzer = _linguistics.MorphologyAnalyzer(
        use_pos_tagger=False, use_word_segmentation=False
    )
except ImportError:
    pass


class MockTokenizer:
    """Deterministic whitespace-level tokenizer implementing the
    encode/decode surface vncompress compressors rely on."""

    _WORD_RE = re.compile(r"\S+", re.UNICODE)

    def __init__(self):
        self._id_to_word = {}
        self._word_to_id = {}

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [self._register(w) for w in self._WORD_RE.findall(text)]

    def _register(self, word):
        if word not in self._word_to_id:
            new_id = len(self._id_to_word)
            self._word_to_id[word] = new_id
            self._id_to_word[new_id] = word
        return self._word_to_id[word]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self._id_to_word.get(i, "") for i in ids)


@pytest.fixture
def tmp_path():
    """Override pytest's built-in tmp_path: its default base temp dir
    (a numbered pytest-of-<user> tree under %TEMP%) is not writable in
    this sandboxed environment (PermissionError scanning old runs).
    Uses a fresh tempfile.mkdtemp() directory instead, which sidesteps
    pytest's own bookkeeping directory entirely."""
    d = tempfile.mkdtemp(prefix="vncompress-test-")
    try:
        yield pathlib.Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tokenizer():
    return MockTokenizer()


# A Vietnamese paragraph mixing function words (đã, của, và, những, ở),
# content words, a reduplicative pair (nhè nhẹ) and repeated words, so
# compressors exercise every word class and the tokenizer reuses ids.
VI_TEXT = (
    "Hôm nay chúng tôi đã đi học ở một ngôi trường đại học rất xa nhưng "
    "chúng tôi vẫn đến đúng giờ vì chúng tôi luôn luôn chuẩn bị kỹ càng "
    "và cẩn thận trước khi ra khỏi nhà, gió thổi nhè nhẹ qua khung cửa sổ "
    "của lớp học trong buổi sáng mùa thu."
)

EN_TEXT = (
    "Today we went to a university that is very far away, but we still "
    "arrived on time because we always prepare carefully before leaving "
    "the house."
)


@pytest.fixture
def vi_text():
    return VI_TEXT


@pytest.fixture
def en_text():
    return EN_TEXT


@pytest.fixture
def vi_ids(tokenizer):
    return tokenizer.encode(VI_TEXT, add_special_tokens=False)
