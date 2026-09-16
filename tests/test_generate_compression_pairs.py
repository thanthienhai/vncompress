"""Tests for scripts/generate_compression_pairs.py.

The paths that matter most here are the ones a happy-path run never reaches:
re-prompting, giving up on a sample, collapsing identical ratio levels, and
backing off a 429. Each one exists because of a specific v1 failure, and each
is driven here by a fake teacher whose behaviour the test chooses, rather than
being left to whatever the paid API happens to do.
"""
import importlib.util
import pathlib
import json
import os
import sys
from collections import Counter

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SPEC = importlib.util.spec_from_file_location(
    "generate_compression_pairs",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "generate_compression_pairs.py"),
)
gen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen)


def _api_connection_error(inner):
    """An openai APIConnectionError raised from an httpx failure, as the SDK does."""
    from openai import APIConnectionError
    try:
        try:
            raise inner
        except Exception as cause:
            raise APIConnectionError(request=httpx.Request('POST', 'https://x/v1/chat')) from cause
    except APIConnectionError as exc:
        return exc


class ScriptedTeacher:
    """Returns a prepared reply per call, recording what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.model = 'fake'
        self.base_url = 'http://fake'
        self.temperature = 0.1
        self.max_tokens = 1024
        self.calls = Counter()

    def chat(self, system, user, api_key=None, max_tokens=None, stop=None):
        self.prompts.append(user)
        self.calls['requests'] += 1
        return self.replies.pop(0)


def words(n):
    return ' '.join(f"w{i}" for i in range(n))


class TestCompressToBudget:
    def test_accepts_an_output_inside_the_budget_on_the_first_try(self):
        teacher = ScriptedTeacher([words(100)])
        text, attempts, _, _ = gen.compress_to_budget(
            teacher, "ngữ cảnh", "câu hỏi?", 100, require_extractive=False)
        assert gen.count_tokens(text) == 100
        assert attempts == 1
        assert teacher.calls['requests'] == 1

    def test_reprompts_when_the_first_output_misses(self):
        # v1's teacher was never told it missed. This one is.
        teacher = ScriptedTeacher([words(400), words(98)])
        text, attempts, _, _ = gen.compress_to_budget(
            teacher, "ngữ cảnh", "câu hỏi?", 100, require_extractive=False)
        assert attempts == 2
        assert gen.count_tokens(text) == 98
        assert "dài 400 từ" in teacher.prompts[1]
        assert "rút ngắn" in teacher.prompts[1]

    def test_tells_the_teacher_which_direction_to_move(self):
        teacher = ScriptedTeacher([words(10), words(100)])
        gen.compress_to_budget(teacher, "ngữ cảnh", "câu hỏi?", 100, require_extractive=False)
        assert "bổ sung" in teacher.prompts[1]

    def test_drops_the_sample_after_two_failed_reprompts(self):
        # The whole point of SS3.1: a sample that never meets its budget is
        # dropped, not kept and labelled afterwards.
        teacher = ScriptedTeacher([words(400), words(390), words(380)])
        text, attempts, _, _ = gen.compress_to_budget(
            teacher, "ngữ cảnh", "câu hỏi?", 100, require_extractive=False)
        assert text is None
        assert attempts == gen.MAX_REPROMPTS + 1
        assert teacher.calls['requests'] == 3

    def test_never_calls_more_than_the_reprompt_allowance(self):
        teacher = ScriptedTeacher([words(999)] * 10)
        gen.compress_to_budget(teacher, "ngữ cảnh", "câu hỏi?", 10, require_extractive=False)
        assert teacher.calls['requests'] == gen.MAX_REPROMPTS + 1

    def test_the_prompt_states_the_budget(self):
        teacher = ScriptedTeacher([words(50)])
        gen.compress_to_budget(teacher, "đoạn văn dài", "câu hỏi?", 50, require_extractive=False)
        assert "50 từ" in teacher.prompts[0]
        assert "câu hỏi?" in teacher.prompts[0]      # SS3.4: query-conditioned
        assert "đoạn văn dài" in teacher.prompts[0]


class TestAnswerability:
    def test_a_compression_that_preserves_the_answer_passes(self):
        teacher = ScriptedTeacher(["Lâm Bá Kiệt"])
        answerable, f1, predicted, method = gen.judge_answerability(
            teacher, "bản nén có đáp án", "Tên gọi nào?", "Lâm Bá Kiệt")
        assert answerable is True
        assert f1 == pytest.approx(1.0)
        assert predicted == "Lâm Bá Kiệt"
        assert method == 'f1'

    def test_a_compression_that_lost_the_answer_fails(self):
        teacher = ScriptedTeacher(["KHÔNG CÓ"])
        answerable, f1, _, method = gen.judge_answerability(
            teacher, "bản nén không liên quan", "Tên gọi nào?", "Lâm Bá Kiệt")
        assert answerable is False
        assert f1 < gen.ANSWERABLE_F1
        assert method == 'none'

    def test_the_judge_only_sees_the_compression(self):
        # If the original context leaked into the judge's prompt it would
        # answer from that and every compression would look answerable.
        teacher = ScriptedTeacher(["đáp án"])
        gen.judge_answerability(teacher, "CHỈ_BẢN_NÉN", "câu hỏi?", "đáp án")
        assert "CHỈ_BẢN_NÉN" in teacher.prompts[0]


class TestAnswerabilityIsNotJustF1:
    """The 50.4% false-negative rate measured on the first v2 run.

    Each case here is a real (gold, judge answer) pair from
    extras/compression_unanswerable.jsonl -- rows the F1-only gate discarded
    while the compression still contained the answer.
    """

    def test_a_more_complete_answer_than_the_gold_is_accepted(self):
        # viquad:Ả_Rập_Xê_Út, F1 0.4211 under the old gate.
        teacher = ScriptedTeacher([
            "Quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị."])
        answerable, f1, _, method = gen.judge_answerability(
            teacher, "Ả Rập Xê Út theo chế độ quân chủ chuyên chế.",
            "Chế độ nhà nước của Ả Rập Xê Út là gì?", "quân chủ chuyên chế")
        assert f1 < gen.ANSWERABLE_F1        # the old gate still rejects it
        assert answerable is True            # ...and it is still correct
        assert method == 'containment'

    def test_a_shorter_answer_than_the_gold_is_accepted(self):
        # viquad:Lưu_vực, F1 0.3636 under the old gate.
        teacher = ScriptedTeacher(["Biển Caspian"])
        answerable, f1, _, method = gen.judge_answerability(
            teacher, "Nước đổ vào biển Caspian, biển Aral và nhiều hồ nhỏ hơn.",
            "Lòng chảo nội lục của châu Á đổ vào đâu nhiều nhất?",
            "biển Caspian, biển Aral và nhiều hồ nhỏ hơn")
        assert f1 < gen.ANSWERABLE_F1
        assert answerable is True
        assert method == 'containment'

    def test_the_answer_span_surviving_verbatim_is_enough(self):
        # The judge wandered off, but the compression provably still carries
        # the span -- the strictest evidence, and it needs no judge at all.
        teacher = ScriptedTeacher(["Tôi không chắc chắn về điều này."])
        answerable, _, _, method = gen.judge_answerability(
            teacher, "Thủ đô của Hà Lan là Amsterdam theo hiến pháp.",
            "Thủ đô của Hà Lan là thành phố nào?", "Amsterdam")
        assert answerable is True
        assert method == 'span'

    def test_a_genuinely_lost_answer_is_still_rejected(self):
        # The gate must stay a gate: nothing here contains the answer.
        teacher = ScriptedTeacher(["KHÔNG CÓ"])
        answerable, _, _, method = gen.judge_answerability(
            teacher, "Bản nén nói về khí hậu và địa hình.",
            "Thủ đô của Hà Lan là thành phố nào?", "Amsterdam")
        assert answerable is False
        assert method == 'none'

    def test_a_very_short_gold_does_not_match_by_coincidence(self):
        # "Bá" inside "Lâm Bá Kiệt" is substring luck, not a preserved answer.
        assert gen.answerability_verdict("Lâm Bá Kiệt", "Bá", "ngữ cảnh", 0.0) == (False, 'none')


# Retry, backoff and give-up used to live inside Teacher.chat and were tested
# here. They now belong to vncompress/api_pool.py, and the tests moved with them
# (tests/test_api_pool.py::TestRunPool). Keeping a second backoff inside the
# client would have been worse than duplicated code: an in-call retry swallows
# the 429 before the pool sees it, so the key being rate-limited never cools
# down and keeps being handed work -- a four-key pool quietly degrading to one
# slow key with nothing in the logs to say so.


class TestCheckpoint:
    def test_resumes_from_an_existing_checkpoint(self, tmp_path):
        path = tmp_path / "checkpoint.jsonl"
        path.write_text(
            json.dumps({'source_id': 'qa_1', 'row': {'id': 'c1'}}) + "\n"
            + json.dumps({'source_id': 'qa_2', 'row': {'id': 'c2'}}) + "\n",
            encoding='utf-8')
        done, rows = gen.load_checkpoint(str(path), dry_run=False)
        assert done == {'qa_1', 'qa_2'}
        assert [r['row']['id'] for r in rows] == ['c1', 'c2']

    def test_missing_checkpoint_is_not_an_error(self, tmp_path):
        done, rows = gen.load_checkpoint(str(tmp_path / "nope.jsonl"), dry_run=False)
        assert done == set() and rows == []


class TestDryRun:
    def test_dry_run_needs_no_api_key(self):
        teacher = gen.Teacher('m', 'http://x', dry_run=True)
        reply = teacher.chat(gen.COMPRESS_SYSTEM, gen.COMPRESS_PROMPT.format(
            target_tokens=5, query="câu hỏi?", context=words(100)))
        assert gen.count_tokens(reply) == 5

    def test_a_real_run_without_a_key_refuses_instead_of_running(self):
        # Key handling moved to the pool: a keyless real run is refused by
        # build_pool, not by the Teacher, which no longer holds a key at all.
        args = type('A', (), {'api_key': None, 'rpm': 50, 'ccu_per_key': 10})
        with pytest.raises(SystemExit):
            gen.build_pool(args, prefix='VNCOMPRESS_NO_SUCH_KEY_PREFIX')


class TestBuildRow:
    def test_records_merged_levels_and_recomputed_metrics(self):
        teacher = gen.Teacher('GLM-5.2', 'http://x', dry_run=True)
        source = {'id': 'qa_1', 'doc_id': 'd1', 'source': 'uit-viquad-2.0',
                  'domain': 'history', 'split': 'train',
                  'context': words(100), 'query': 'câu hỏi?', 'answer': 'w1'}
        # One ratio per row now; levels are merged in deduplicate(), not here.
        row = gen.build_row(source, 2.0, words(50), 50, 1, True, 0.9, 'w1', teacher, True)
        assert row['stable_across_ratios'] == [2.0]
        assert row['requested_ratio'] == 2.0
        assert row['realized_tokens'] == 50
        assert row['realized_ratio'] == pytest.approx(2.0)
        assert row['budget_compliance'] is True
        assert row['answerable_from_compression'] is True

    def test_carries_the_fidelity_and_provenance_fields(self):
        """The columns the shipped v2 rows had no way to be checked against.

        Without `sentence_extractive_ratio` a rewrite passes as a compression;
        without `judge_model` a reader cannot tell the teacher graded itself.
        """
        teacher = gen.Teacher('GLM-5.2', 'http://x', dry_run=True)
        context = "Hà Nội là thủ đô của Việt Nam. Thành phố nằm bên sông Hồng."
        source = {'id': 'qa_1', 'doc_id': 'd1', 'source': 's', 'domain': 'other',
                  'split': 'train', 'context': context, 'query': 'q', 'answer': 'Hà Nội'}
        quoted = gen.build_row(source, 2.0, "Hà Nội là thủ đô của Việt Nam.", 6, 1,
                               True, 1.0, 'Hà Nội', teacher, True, method='f1')
        assert quoted['sentence_extractive_ratio'] == pytest.approx(1.0)
        assert quoted['unsupported_numbers'] is None      # no numbers to check
        assert quoted['answerability_method'] == 'f1'
        assert quoted['verification_method'] == 'none'
        assert quoted['gen_config']['judge_model'] == 'GLM-5.2'
        assert quoted['gen_config']['judge_is_teacher'] is True

    def test_a_rewritten_gold_is_visible_in_the_metrics(self):
        """The v2 failure, reproduced small: familiar words, invented facts."""
        teacher = gen.Teacher('GLM-5.2', 'http://x', dry_run=True)
        context = "Hà Nội là thủ đô của Việt Nam. Thành phố nằm bên sông Hồng."
        source = {'id': 'qa_2', 'doc_id': 'd1', 'source': 's', 'domain': 'other',
                  'split': 'train', 'context': context, 'query': 'q', 'answer': 'Hà Nội'}
        # Almost every token is lifted from the context, no sentence is, and
        # the year 1010 appears nowhere in the source.
        rewritten = gen.build_row(
            source, 2.0, "Thành phố Hà Nội là thủ đô của Việt Nam, nằm bên sông Hồng năm 1010.",
            12, 1, True, 1.0, 'Hà Nội', teacher, True, method='f1')
        # Token overlap is reassuring and sentence overlap is zero -- that gap
        # is the whole point of carrying both.
        assert rewritten['extractive_ratio'] > 0.7
        assert rewritten['sentence_extractive_ratio'] == 0.0
        assert rewritten['unsupported_numbers'] == ['1010']

    def test_a_row_over_budget_is_marked_non_compliant(self):
        teacher = gen.Teacher('GLM-5.2', 'http://x', dry_run=True)
        source = {'id': 'qa_1', 'doc_id': 'd1', 'source': 's', 'domain': 'other',
                  'split': 'train', 'context': words(100), 'query': 'q', 'answer': 'a'}
        row = gen.build_row(source, 8.0, words(90), 12, 3, False, 0.1, '', teacher, True)
        assert row['budget_compliance'] is False


class TestGlobalDeduplication:
    """B2/P2: (doc_id, gold_compression) is unique run-wide, not per source row.

    Collapsing only inside one source row -- all the per-row `by_text` map can
    do -- misses the common case: two different questions about the same
    document whose answers live in the same passage converge on the same
    compression. Measured on the v2 eval set, 271 of 1,000 rows share a
    (doc_id, gold_compression) with another row.
    """

    def _row(self, row_id, doc_id, gold, ratios):
        return {'id': row_id, 'doc_id': doc_id, 'gold_compression': gold,
                'requested_ratio': min(ratios), 'stable_across_ratios': list(ratios),
                'answerable_from_compression': True}

    def test_duplicates_across_different_source_rows_collapse(self):
        stats = Counter()
        rows = gen.deduplicate([
            self._row('a', 'doc1', 'Hà Nội là thủ đô.', [2.0]),
            self._row('b', 'doc1', 'Hà Nội là thủ đô.', [4.0]),
        ], stats)
        assert len(rows) == 1
        assert stats['rows merged (duplicate across source rows)'] == 1

    def test_the_merged_row_keeps_every_ratio_that_produced_it(self):
        # Otherwise collapsing loses the fact that 4x returned the same text --
        # which is the thing SS3.2 asks to be recorded, not discarded.
        rows = gen.deduplicate([
            self._row('a', 'doc1', 'Hà Nội là thủ đô.', [2.0]),
            self._row('b', 'doc1', 'Hà Nội là thủ đô.', [4.0, 8.0]),
        ], Counter())
        assert rows[0]['stable_across_ratios'] == [2.0, 4.0, 8.0]
        assert rows[0]['requested_ratio'] == 2.0

    def test_whitespace_variants_are_the_same_row(self):
        rows = gen.deduplicate([
            self._row('a', 'doc1', 'Hà Nội  là\nthủ đô.', [2.0]),
            self._row('b', 'doc1', ' Hà Nội là thủ  đô. ', [4.0]),
        ], Counter())
        assert len(rows) == 1

    def test_the_same_text_under_a_different_document_is_not_a_duplicate(self):
        # The key is (doc_id, text). Two documents may legitimately share a
        # boilerplate sentence, and merging them would lose a real row.
        rows = gen.deduplicate([
            self._row('a', 'doc1', 'Điều 1. Phạm vi điều chỉnh.', [2.0]),
            self._row('b', 'doc2', 'Điều 1. Phạm vi điều chỉnh.', [2.0]),
        ], Counter())
        assert len(rows) == 2

    def test_distinct_compressions_all_survive(self):
        rows = gen.deduplicate([
            self._row('a', 'doc1', 'Bản nén dài hơn nhiều chữ.', [2.0]),
            self._row('b', 'doc1', 'Bản nén ngắn.', [8.0]),
        ], Counter())
        assert len(rows) == 2


class TestCheckpointIsolation:
    """A --dry-run must not be able to spend or poison a real run.

    It could: both wrote the same checkpoint, so a rehearsal left its fabricated
    rows behind and the next real run opened with "Resuming: 3 source row(s)
    already processed" and shipped them.
    """

    def test_the_two_modes_use_different_files(self):
        real = gen.checkpoint_path_for('/data/x', dry_run=False)
        fake = gen.checkpoint_path_for('/data/x', dry_run=True)
        assert real != fake and 'dryrun' in fake and 'dryrun' not in os.path.basename(real)

    def test_a_real_run_ignores_dry_run_records(self, tmp_path):
        path = tmp_path / 'ckpt.jsonl'
        path.write_text(json.dumps({'source_id': 's1', 'dry_run': True,
                                    'row': {'id': 'fake'}}) + "\n", encoding='utf-8')
        done, rows = gen.load_checkpoint(str(path), dry_run=False)
        assert done == set() and rows == []

    def test_a_dry_run_ignores_real_records(self, tmp_path):
        path = tmp_path / 'ckpt.jsonl'
        path.write_text(json.dumps({'source_id': 's1', 'dry_run': False,
                                    'row': {'id': 'real'}}) + "\n", encoding='utf-8')
        done, rows = gen.load_checkpoint(str(path), dry_run=True)
        assert done == set() and rows == []

    def test_matching_records_resume_normally(self, tmp_path):
        path = tmp_path / 'ckpt.jsonl'
        path.write_text(json.dumps({'source_id': 's1', 'dry_run': False,
                                    'row': {'id': 'real'}}) + "\n", encoding='utf-8')
        done, rows = gen.load_checkpoint(str(path), dry_run=False)
        assert done == {'s1'} and len(rows) == 1

    def test_a_legacy_checkpoint_without_the_flag_reads_as_a_real_run(self, tmp_path):
        # Files written before the flag existed came from real runs.
        path = tmp_path / 'ckpt.jsonl'
        path.write_text(json.dumps({'source_id': 's1', 'row': {'id': 'legacy'}}) + "\n",
                        encoding='utf-8')
        assert gen.load_checkpoint(str(path), dry_run=False)[0] == {'s1'}
        assert gen.load_checkpoint(str(path), dry_run=True)[0] == set()


class TestBatchShards:
    """Per-batch durability: a 15-hour run must not lose everything at hour 14."""

    def _row(self, row_id, doc_id, gold, ratio, answerable=True):
        return {'id': row_id, 'doc_id': doc_id, 'gold_compression': gold,
                'requested_ratio': ratio, 'stable_across_ratios': [ratio],
                'answerable_from_compression': answerable}

    def test_a_shard_round_trips(self, tmp_path):
        rows = [self._row('a', 'd1', 'nén một', 2.0), self._row('b', 'd1', 'nén hai', 4.0)]
        gen.write_shard(str(tmp_path), 0, rows)
        assert gen.read_shards(str(tmp_path)) == rows

    def test_shards_are_read_in_batch_order(self, tmp_path):
        gen.write_shard(str(tmp_path), 2, [self._row('c', 'd', 'ba', 2.0)])
        gen.write_shard(str(tmp_path), 0, [self._row('a', 'd', 'mot', 2.0)])
        gen.write_shard(str(tmp_path), 1, [self._row('b', 'd', 'hai', 2.0)])
        assert [r['id'] for r in gen.read_shards(str(tmp_path))] == ['a', 'b', 'c']

    def test_merge_deduplicates_ACROSS_shards(self, tmp_path):
        # The duplicate this design is most likely to produce: two questions
        # about one document, landing in different batches, compressed to the
        # same text. Per-shard dedup cannot see it; only the merge can.
        gen.write_shard(str(tmp_path), 0, [self._row('a', 'd1', 'Hà Nội là thủ đô.', 2.0)])
        gen.write_shard(str(tmp_path), 1, [self._row('b', 'd1', 'Hà Nội là thủ đô.', 4.0)])
        keep, _ = gen.merge_shards(str(tmp_path), str(tmp_path / 'out'))
        assert len(keep) == 1
        assert keep[0]['stable_across_ratios'] == [2.0, 4.0]

    def test_merge_splits_answerable_from_not(self, tmp_path):
        gen.write_shard(str(tmp_path), 0, [
            self._row('a', 'd1', 'giữ lại', 2.0, answerable=True),
            self._row('b', 'd2', 'loại ra', 2.0, answerable=False)])
        keep, reject = gen.merge_shards(str(tmp_path), str(tmp_path / 'out'))
        assert [r['id'] for r in keep] == ['a']
        assert [r['id'] for r in reject] == ['b']
        out = tmp_path / 'out'
        assert (out / 'compression.jsonl').exists()
        assert (out / 'extras' / 'compression_unanswerable.jsonl').exists()

    def test_merging_twice_is_not_cumulative(self, tmp_path):
        # merge_shards runs after EVERY batch, so it must rewrite rather than
        # append. Appending would multiply the dataset by the batch count.
        gen.write_shard(str(tmp_path), 0, [self._row('a', 'd1', 'nén', 2.0)])
        gen.merge_shards(str(tmp_path), str(tmp_path / 'out'))
        keep, _ = gen.merge_shards(str(tmp_path), str(tmp_path / 'out'))
        assert len(keep) == 1
        lines = (tmp_path / 'out' / 'compression.jsonl').read_text(encoding='utf-8').splitlines()
        assert len(lines) == 1

    def test_merging_with_no_shards_yet_is_empty_not_an_error(self, tmp_path):
        keep, reject = gen.merge_shards(str(tmp_path), str(tmp_path / 'out'))
        assert keep == [] and reject == []

    def test_the_manifest_round_trips(self, tmp_path):
        manifest = {'batches': {'0': {'status': 'done', 'rows': 12}}}
        gen.save_manifest(str(tmp_path), manifest)
        assert gen.load_manifest(str(tmp_path)) == manifest

    def test_a_missing_manifest_reads_as_nothing_done(self, tmp_path):
        # Not an error: the first run of a fresh dataset has no manifest, and
        # treating that as a crash would make starting impossible.
        assert gen.load_manifest(str(tmp_path)) == {'batches': {}}

    def test_dry_run_shards_never_touch_the_real_ones(self, tmp_path):
        gen.write_shard(str(tmp_path), 0, [self._row('real', 'd', 'thật', 2.0)], dry_run=False)
        gen.write_shard(str(tmp_path), 0, [self._row('fake', 'd', 'giả', 2.0)], dry_run=True)
        assert [r['id'] for r in gen.read_shards(str(tmp_path), dry_run=False)] == ['real']
        assert [r['id'] for r in gen.read_shards(str(tmp_path), dry_run=True)] == ['fake']
        assert gen.manifest_path(str(tmp_path), True) != gen.manifest_path(str(tmp_path), False)

    def test_a_shard_write_is_atomic(self, tmp_path):
        # os.replace, so a kill mid-write leaves the previous shard intact
        # rather than a half-file the merge would fail to parse.
        gen.write_shard(str(tmp_path), 0, [self._row('a', 'd', 'x', 2.0)])
        assert not list(pathlib.Path(gen.shard_dir(str(tmp_path))).glob('*.tmp'))



class TestRateLimitAccounting:
    """The limiter has to count what the PROVIDER counts: requests."""

    def _teacher_with(self, responses):
        """A Teacher whose transport is scripted, everything else real."""
        from vncompress.api_pool import ApiKey

        teacher = gen.Teacher(model='m', base_url='http://x', dry_run=False)
        key = ApiKey('k', 0, rpm=10 ** 6)
        seq = iter(responses)

        class _Completions:
            @staticmethod
            def create(**kwargs):
                nxt = next(seq)
                if isinstance(nxt, Exception):
                    raise nxt
                return nxt

        class _Client:
            chat = type('c', (), {'completions': _Completions})()

        teacher._clients[key.index] = _Client()
        taken = []
        real = key.limiter.acquire
        key.limiter.acquire = lambda stop=None: (taken.append(1), real(stop))[1]
        return teacher, key, taken

    @staticmethod
    def _ok(text='xin chao'):
        msg = type('m', (), {'content': text})()
        choice = type('c', (), {'message': msg, 'finish_reason': 'stop'})()
        return type('r', (), {'choices': [choice], 'usage': None})()

    def test_a_successful_call_takes_exactly_one_permit(self):
        teacher, key, taken = self._teacher_with([self._ok()])
        teacher.chat('sys', 'user', key)
        assert len(taken) == 1

    def test_each_transport_retry_takes_its_own_permit(self, monkeypatch):
        monkeypatch.setattr(gen, 'TRANSPORT_RETRY_WAIT', 0.0)
        # A retry is another real request. Taking one permit and sending four
        # is the same under-count that let a 150/min limiter admit 600 -- just
        # smaller, and therefore easier to leave in.
        dropped = _api_connection_error(
            httpx.RemoteProtocolError('Server disconnected without sending a response.'))
        teacher, key, taken = self._teacher_with([dropped, dropped, self._ok()])
        teacher.chat('sys', 'user', key)
        assert len(taken) == 3
        assert teacher.calls['requests'] == 3
        assert teacher.calls['transport_retry'] == 2

    def test_a_429_is_not_retried_here_it_goes_to_the_pool(self):
        # The pool owns cooling the key. A retry that swallows the 429 leaves a
        # rate-limited key in rotation, which is the whole reason for the rule.
        class _RateLimited(Exception):
            status_code = 429

        teacher, key, taken = self._teacher_with([_RateLimited(), self._ok()])
        with pytest.raises(_RateLimited):
            teacher.chat('sys', 'user', key)
        assert len(taken) == 1
        assert teacher.calls['transport_retry'] == 0

    def test_it_gives_up_after_the_allowance_rather_than_looping(self, monkeypatch):
        monkeypatch.setattr(gen, 'TRANSPORT_RETRY_WAIT', 0.0)
        dropped = _api_connection_error(httpx.ReadError('died'))
        teacher, key, taken = self._teacher_with([dropped] * (gen.TRANSPORT_RETRIES + 1))
        with pytest.raises(Exception):
            teacher.chat('sys', 'user', key)
        assert len(taken) == gen.TRANSPORT_RETRIES + 1


CONTEXT_3 = ("Hà Nội là thủ đô của Việt Nam. "
             "Thành phố nằm bên bờ sông Hồng. "
             "Dân số khoảng tám triệu người.")


class TestExtractiveGate:
    """SS3.1 v3: the gold must be SELECTED from the context, not written.

    v2 had no such gate and shipped 973 rows with a median
    sentence_extractive_ratio of 0.038.
    """

    def test_quoted_sentences_pass(self):
        assert gen.extractive_problem(CONTEXT_3, "Hà Nội là thủ đô của Việt Nam.") is None

    def test_a_rewrite_is_named_as_a_rewrite(self):
        problem = gen.extractive_problem(
            CONTEXT_3, "Thủ đô của Việt Nam chính là thành phố Hà Nội bên sông Hồng.")
        assert problem and 'nguyên văn' in problem

    def test_the_offending_sentence_is_quoted_back(self):
        # A generic "try again" earns a generic retry; the model has to be told
        # which sentence it invented.
        problem = gen.extractive_problem(CONTEXT_3, "Thủ đô của Việt Nam là Hà Nội.")
        assert 'Thủ đô của Việt Nam là Hà Nội.' in problem

    def test_an_invented_number_is_caught_even_when_quoting(self):
        # Every sentence here is verbatim except the appended year, which is
        # the v2 failure that token overlap could never see.
        gold = "Hà Nội là thủ đô của Việt Nam. Thành phố được lập năm 1010."
        problem = gen.extractive_problem(CONTEXT_3, gold)
        assert problem and '1010' in problem

    def test_empty_output_is_not_an_extractive_problem(self):
        # Nothing to judge: that is a budget failure, handled separately.
        assert gen.extractive_problem(CONTEXT_3, "") is None


class TestCompressToSpec:
    def test_a_rewrite_is_re_prompted_then_dropped(self):
        rewrite = "Thủ đô của Việt Nam chính là thành phố Hà Nội bên sông Hồng."
        teacher = ScriptedTeacher([rewrite] * 5)
        text, attempts, _, problem = gen.compress_to_budget(
            teacher, CONTEXT_3, "Thủ đô là gì?", len(rewrite.split()))
        assert text is None                       # dropped, not kept and labelled
        assert attempts == gen.MAX_REPROMPTS + 1
        assert problem and 'nguyên văn' in problem

    def test_a_model_that_corrects_itself_is_kept(self):
        rewrite = "Thủ đô của Việt Nam chính là Hà Nội."
        quoted = "Hà Nội là thủ đô của Việt Nam."
        teacher = ScriptedTeacher([rewrite, quoted])
        text, attempts, _, problem = gen.compress_to_budget(
            teacher, CONTEXT_3, "Thủ đô là gì?", len(quoted.split()))
        assert text == quoted
        assert attempts == 2
        assert problem is None

    def test_the_re_prompt_states_the_extractive_rule(self):
        rewrite = "Thủ đô của Việt Nam chính là Hà Nội."
        teacher = ScriptedTeacher([rewrite] * 5)
        gen.compress_to_budget(teacher, CONTEXT_3, "Thủ đô là gì?", len(rewrite.split()))
        assert 'NGUYÊN VĂN' in teacher.prompts[1]

    def test_the_gate_can_be_turned_off_on_purpose(self):
        rewrite = "Thủ đô của Việt Nam chính là Hà Nội."
        teacher = ScriptedTeacher([rewrite])
        text, _, _, problem = gen.compress_to_budget(
            teacher, CONTEXT_3, "Thủ đô là gì?", len(rewrite.split()),
            require_extractive=False)
        assert text == rewrite
        assert problem is None

    def test_the_prompt_demands_verbatim_selection(self):
        teacher = ScriptedTeacher(["Hà Nội là thủ đô của Việt Nam."])
        gen.compress_to_budget(teacher, CONTEXT_3, "Thủ đô là gì?", 7)
        assert 'NGUYÊN VĂN' in teacher.prompts[0]
        assert 'KHÔNG viết lại' in teacher.prompts[0]


class TestSyntheticQaIsOptIn:
    """Stage 2b output is read only when asked for, and never on the test split."""

    def _write(self, tmp_path, name, rows):
        with open(tmp_path / name, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    HUMAN = {'id': 'qa_1', 'split': 'train', 'query': 'q?', 'answer': 'a',
             'doc_id': 'viquad:D', 'context': 'c', 'question_source': 'human'}
    SYNTHETIC = {'id': 'qa_syn_1', 'split': 'train', 'query': 'q2?', 'answer': 'a2',
                 'doc_id': 'uvw:E', 'context': 'c', 'question_source': 'teacher-generated'}

    def test_default_reads_only_the_human_file(self, tmp_path):
        self._write(tmp_path, 'qa.jsonl', [self.HUMAN])
        self._write(tmp_path, 'qa_synthetic.jsonl', [self.SYNTHETIC])
        rows = gen.load_sources(str(tmp_path), {'train'})
        assert [r['id'] for r in rows] == ['qa_1']

    def test_opting_in_merges_both(self, tmp_path):
        self._write(tmp_path, 'qa.jsonl', [self.HUMAN])
        self._write(tmp_path, 'qa_synthetic.jsonl', [self.SYNTHETIC])
        rows = gen.load_sources(str(tmp_path), {'train'}, include_synthetic=True)
        assert {r['id'] for r in rows} == {'qa_1', 'qa_syn_1'}

    def test_opting_in_without_the_file_is_an_error(self, tmp_path):
        # Silently running on human rows only, after being asked for both,
        # produces a result whose training set nobody can reconstruct.
        self._write(tmp_path, 'qa.jsonl', [self.HUMAN])
        with pytest.raises(SystemExit, match='qa_synthetic'):
            gen.load_sources(str(tmp_path), {'train'}, include_synthetic=True)

    def test_a_synthetic_row_on_the_test_split_stops_the_run(self, tmp_path):
        """B5/E3: teacher questions must never reach the independent test set."""
        self._write(tmp_path, 'qa.jsonl', [self.HUMAN])
        self._write(tmp_path, 'qa_synthetic.jsonl',
                    [dict(self.SYNTHETIC, split='test')])
        with pytest.raises(SystemExit, match='TEST split'):
            gen.load_sources(str(tmp_path), {'train'}, include_synthetic=True)
