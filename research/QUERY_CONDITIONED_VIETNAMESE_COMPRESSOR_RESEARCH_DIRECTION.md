# Research Direction — Query-Conditioned Vietnamese Compressor with Linguistic Supervision

**Project:** VNCompress / LACC  
**Status:** Proposed research direction after Wave 1  
**Date:** 2026-09-14  
**Purpose:** Consolidate the empirical findings, literature evidence, research-gap analysis, methodological redesign, training proposal, benchmark plan, ablations, and go/no-go criteria for the next version of the paper.

> **Core recommendation:** Do not continue the paper as “tone-aware Vietnamese prompt compression”. Reframe it as a **query-conditioned Vietnamese context compressor with linguistically supervised and dependency-aware selection**. Tone should remain as a controlled/negative linguistic-supervision experiment rather than the primary compression signal.

---

## 1. Executive conclusion

Wave 1 has produced a useful negative scientific result: the Vietnamese linguistic property selected as the central signal — tone — can be learned by the model, but learning it does not make the compressor better at the downstream task. The current LACC implementation is fundamentally query-agnostic: perplexity, tone, and morphology are scored from the context, while the query enters only through a late, shallow keyword-match multiplier. This is a plausible root cause of the observed failure to retain answer-bearing content.

The next research version should therefore make **task relevance the primary objective** and use Vietnamese linguistic structure as **auxiliary supervision/inductive bias** rather than as a direct replacement for relevance.

The proposed research question is:

> **Can a lightweight query-conditioned compressor for Vietnamese long contexts improve downstream task utility by jointly learning query relevance and Vietnamese linguistic/structural supervision, while preserving the evidence dependencies required for faithful answers?**

The strongest proposed direction is a three-layer design:

```text
Question + Context
        |
        v
Query-conditioned Vietnamese encoder
        |
        +-------------------+
        |                   |
        v                   v
Task relevance       Linguistic supervision
(main objective)     POS / NER / DEP / morphology / tone
        |                   |
        +---------+---------+
                  v
       Structural/evidence constraints
          dependency closure
                  |
                  v
       Sentence -> Clause -> Token
             compression
                  |
                  v
          Compressed context
                  |
                  v
              Target LLM
```

This direction is stronger than simply replacing the tone probe with a relevance probe because it directly addresses two distinct failure modes:

1. **Relevance failure:** query-agnostic scoring does not know which content answers the question.
2. **Completeness failure:** independent token selection can keep answer-bearing fragments while deleting antecedents, bridge facts, or syntactic/discourse support.

---

# 2. What Wave 1 actually established

The Wave 1 technical report concludes that the central tone-aware hypothesis does not hold. Tone-aware compression preserved **93.2% of marked-tone tokens**, but recovered only **2.3% of needle information at 2× compression**, whereas a pure perplexity scorer recovered **74.2%**. On extracted QA, tone-aware compression could fall below random dropout. A controlled A/B experiment also found that the learned tone probe preserved tone worse than the rule-based signal (ΔTPR −0.246, CI [−0.251, −0.241]) while producing better task answers (Δtoken-F1 +0.086, CI [+0.028, +0.146], win-rate 76%).

At the same time, the training method itself was not shown to be broken. The 2×2 probe control study found that LoRA + phonological consistency supervision added **+0.0625 macro-F1** on the real tone task and **−0.0147** on the token-identity control, indicating tone-specific learning.

The correct interpretation is therefore:

```text
The model learns Vietnamese linguistic information
                 |
                 v
        Tone representation is real
                 |
                 v
But tone is not the same as task relevance
                 |
                 v
Tone-aware token importance is the wrong compression objective
```

This distinction should become a central methodological lesson in the paper rather than being hidden as a failed experiment.

**Source:** `results/report/wave_1/2026-09-05_lacc-technical-report.md`.

---

# 3. Root-cause diagnosis of the current LACC design

The current Wave 2 proposal document identifies three implementation-level findings that materially change how the original paper should be interpreted.

## 3.1 Query-agnostic scoring is the main conceptual weakness

The three main LACC scores are computed only from the context. The question does not participate in the core scoring function. It enters only once through a shallow keyword-overlap multiplier after the main scores have already been computed.

Current conceptual pipeline:

```text
Context
  |
  +--> PPL
  +--> Tone
  +--> Morphology
          |
          v
       Score blend
          |
          v
       Top-K tokens
          |
          +---- late keyword multiplier from query
```

Desired pipeline:

```text
Question + Context
        |
        v
Query-conditioned representation
        |
        v
Task relevance
        |
        v
Keep/drop decision
```

This is especially important for needle-in-haystack and QA: the compressor needs to know **what the user is looking for**, not merely which tokens are linguistically salient or surprising.

**Source:** `research/wave2_proposals.md`.

## 3.2 Token-level global top-K creates fragmentation

The current selector effectively uses global token ranking inside the protected boundary region. This can preserve isolated high-scoring tokens while removing the sentence/clause or support context needed to interpret them.

Example:

```text
A: Nguyễn Văn A sinh tại Nam Định.
B: Nam Định thuộc vùng Đồng bằng sông Hồng.

Q: Nguyễn Văn A sinh ở vùng nào?

Naive token selector:
    KEEP: Đồng bằng sông Hồng
    DROP: A

Required evidence:
    A + B
```

Therefore the next design should move from independent token ranking toward **hierarchical selection plus structural restoration**.

## 3.3 Additive blending can dilute a strong signal

Wave 1 found that the pure perplexity arm could outperform the combined blend. The proposal therefore should not simply increase the number of signals in:

```text
S = w1 * PPL + w2 * tone + w3 * morphology
```

Instead, relevance should dominate, while linguistic signals act as auxiliary constraints or calibrated modifiers.

## 3.4 Scorer size is not automatically an advantage

The Wave 1 report records a notable mismatch: the trained Qwen3-4B scorer was more expensive and nevertheless underperformed the off-the-shelf Qwen2.5-0.5B scorer on the relevant compression measurement. This suggests that the scorer should be optimized for **selection quality per unit cost**, not parameter count.

This motivates an encoder-based compressor, e.g. PhoBERT-base, as the main experimental model rather than continuing to increase the size of the generative scorer.

---

# 4. Revised research question and paper positioning

## 4.1 Recommended research question

> **Can query-conditioned compression for Vietnamese long-context tasks be improved by explicit linguistic supervision and structural evidence constraints?**

## 4.2 Recommended working title

Primary option:

> **Query-Conditioned Vietnamese Context Compression with Linguistic and Structural Supervision**

Alternative:

> **Linguistically-Supervised Query-Conditioned Compression for Vietnamese Long Contexts**

Implementation/research continuity name:

> **LACC-2 — Query-Conditioned Linguistic-Aware Context Compression**

The phrase **“linguistically supervised”** is preferable to merely “linguistic-aware” because the intended contribution is learned auxiliary supervision rather than a collection of hand-written rules.

## 4.3 What should NOT be claimed

Do not claim:

- “the first linguistic-aware prompt compressor”;
- “tone is necessary for Vietnamese compression”;
- “Vietnamese compression is fundamentally solved by tone preservation”;
- “PhoBERT + token classification is novel by itself”.

Existing work already covers linguistic-rule-based compression, token-classification compression, and query-aware compression. The novelty should instead be framed around their **intersection**:

```text
Query-conditioned relevance
        +
Vietnamese linguistic supervision
        +
Evidence/dependency completeness
        +
Vietnamese-specific evaluation
```

---

# 5. Literature landscape and research gap

## 5.1 LLMLingua — general prompt compression

**Jiang et al., “LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models”, EMNLP 2023.**

Reference: https://arxiv.org/abs/2310.05736

LLMLingua established perplexity-guided coarse-to-fine prompt compression. It is an essential baseline, but the basic scoring is not explicitly query-conditioned.

**Implication for VNCompress:** PPL is a strong baseline, but PPL alone does not directly optimize answer relevance.

## 5.2 LongLLMLingua — query-aware compression

**Jiang et al., “LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression”, ACL 2024.**

Reference: https://aclanthology.org/2024.acl-long.91/

LongLLMLingua introduces question-aware coarse-grained compression and contrastive perplexity. The paper reports substantial gains in long-context QA at high compression ratios, including up to 21.4% improvement on NaturalQuestions at around 4× compression in its reported setting.

**Implication:** Query conditioning is not an optional cosmetic feature. It is a central direction that the original LACC design does not exploit deeply.

## 5.3 LLMLingua-2 — learned token classification

**Pan et al., “LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression”, Findings of ACL 2024.**

Reference: https://aclanthology.org/2024.findings-acl.57/

LLMLingua-2 distills compression decisions from a stronger teacher into an encoder-based token classification model. It demonstrates that direct learned keep/drop prediction can be much faster than generative scoring.

**Implication:** A PhoBERT-based compressor is technically justified, but simply reproducing LLMLingua-2 for Vietnamese is insufficient novelty. The Vietnamese linguistic supervision and structural objectives must provide the research distinction.

## 5.4 QuerySelect / Adaptive QuerySelect

**“Fundamental Limits of Prompt Compression” (NeurIPS 2024).**

Reference: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html

This work formulates prompt compression using a rate-distortion perspective and investigates query-aware selection, including QuerySelect and Adaptive QuerySelect. It reinforces the importance of conditioning selection on the query.

**Implication:** Query-conditioned compression itself is not the novelty. It should be the foundation on which the Vietnamese-specific contribution is tested.

## 5.5 PartPrompt — linguistic structure in compression

**“PartPrompt: A Part-of-Speech Guided Prompt Compression Method”.**

Reference: https://arxiv.org/abs/2409.15395

PartPrompt uses syntactic/linguistic structure and entropy to guide pruning. This demonstrates that linguistic information can be useful in prompt compression.

**Implication:** The paper must not claim that using linguistic structure alone is novel. The stronger gap is **learned Vietnamese linguistic supervision inside a query-conditioned compressor**.

## 5.6 Recent linguistic-rule compression

**“Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors” (2026).**

Reference: https://arxiv.org/abs/2607.25335

This recent work combines lexical, syntactic, semantic and discourse rules and explores them as effective prompt compressors without relying on expensive LM scoring at inference.

**Implication:** “linguistic-aware compression” is now a populated direction. VNCompress should emphasize **learned supervision, Vietnamese-specific language structure, query conditioning, and structural completeness** rather than hand-written linguistic rules alone.

## 5.7 Referential dangling / structural completeness

**“Referential Dangling in Context Compression” (2026).**

Reference: https://arxiv.org/abs/2608.04569

This work identifies a failure mode where a compressor retains an answer-bearing fragment but removes the antecedent or bridge information needed to interpret it. Reported dangling rates are substantial for some compressors and restoration of supporting context can improve downstream accuracy under the same budget.

**Implication:** This is highly relevant to the proposed dependency/evidence-closure component. Relevance alone is not sufficient; the retained context must be **structurally complete enough to support the answer**.

## 5.8 Vietnamese language-model foundations

### PhoBERT

**Nguyen & Nguyen, “PhoBERT: Pre-trained language models for Vietnamese”, Findings of EMNLP 2020.**

Reference: https://aclanthology.org/2020.findings-emnlp.92/

PhoBERT provides a strong Vietnamese-specific encoder and has demonstrated effectiveness across POS, dependency parsing, NER and other Vietnamese NLP tasks.

### VnCoreNLP

**Nguyen et al., “VnCoreNLP: A Vietnamese Natural Language Processing Toolkit”, NAACL 2018 Demonstrations.**

Reference: https://aclanthology.org/N18-5012/

VnCoreNLP provides Vietnamese word segmentation, POS tagging, NER and dependency parsing.

**Implication:** These resources make explicit Vietnamese linguistic supervision practical without requiring a large generative model.

---

# 6. The actual research gap

The literature suggests the following matrix:

| Dimension | Existing literature | VNCompress opportunity |
|---|---|---|
| Query-aware compression | LongLLMLingua, QuerySelect | Baseline/foundation |
| Learned token compression | LLMLingua-2 | Baseline/foundation |
| Linguistic-rule compression | PartPrompt, recent 2026 rule-based work | Not sufficient as novelty |
| Vietnamese pretrained encoder | PhoBERT and related models | Infrastructure |
| Vietnamese-specific compression | Relatively underexplored | Strong application/research setting |
| Query-conditioned + Vietnamese linguistic supervision | Limited/unclear | **Primary gap** |
| Evidence/dependency completeness in Vietnamese compression | Very limited | **Secondary gap** |
| Vietnamese compression benchmark with linguistic stress tests | Limited | **Benchmark contribution** |

The key conceptual distinction is:

```text
linguistic importance != task relevance

task relevance != structural completeness
```

Therefore the proposed compressor should model all three explicitly:

```text
              Query
                |
                v
        Task relevance
                |
                +----------------+
                |                |
                v                v
       Linguistic validity   Evidence support
                |                |
                +--------+-------+
                         v
                Final compression
```

---

# 7. Proposed model: Query-Conditioned Vietnamese Linguistic Compressor

Working name: **VQLC** or **LACC-2**.

## 7.1 Architecture

```text
                         QUESTION
                            |
                            v
                  +---------------------+
                  | Vietnamese Query    |
                  | Encoder             |
                  +----------+----------+
                             |
                             v
       +-------------------------------------------+
       | Vietnamese Context Encoder                |
       | PhoBERT-base / PhoBERT-large / XLM-R     |
       +--------------------+----------------------+
                            |
            +---------------+----------------+
            |               |                |
            v               v                v
       Relevance       Linguistic        Structural
         head             heads             head
            |          POS/NER/Tone       DEP/support
            |          morphology          closure
            +---------------+----------------+
                            |
                            v
                 Hierarchical selector
                    sentence -> clause
                         -> token
                            |
                            v
                  Dependency/evidence
                       closure
                            |
                            v
                     Budget controller
                            |
                            v
                  COMPRESSED CONTEXT
                            |
                            v
                       Target LLM
                            |
                            v
                          ANSWER
```

## 7.2 Core design principle

**Relevance is the main objective. Linguistic supervision is auxiliary. Structural constraints protect completeness.**

This is deliberately different from the Wave 1 architecture, where tone was used as a direct token-importance signal.

---

# 8. Supervision design

## 8.1 Relevance labels

Do not define the training label simply as:

```text
answer token = 1
all other tokens = 0
```

That creates the wrong learning target because **answer-bearing content is not necessarily the entire evidence set**.

Instead define multiple evidence roles:

```text
KEEP_ANSWER       answer span/token
KEEP_EVIDENCE     directly supports answer
KEEP_SUPPORT      antecedent / bridge / required context
DROP_REDUNDANT    semantically duplicated or unnecessary
DROP_IRRELEVANT   unrelated content
```

The final binary decision can still be:

```text
KEEP = {ANSWER, EVIDENCE, SUPPORT}
DROP = {REDUNDANT, IRRELEVANT}
```

but the richer labels should be available for auxiliary training and analysis.

## 8.2 Teacher-based data distillation

Follow the spirit of LLMLingua-2 but extend the target definition:

```text
Question + Context
        |
        v
Strong teacher LLM
        |
        +--> Answer
        +--> Minimal evidence spans
        +--> Supporting spans
        +--> Distractors
        |
        v
Token/sentence supervision
```

Teacher instruction should require extraction from the **original context**, not rewriting:

> Identify the minimum set of original spans that are sufficient to answer the question. Mark answer-bearing spans, evidence spans, and supporting/bridging spans. Do not rewrite or introduce new information.

## 8.3 Linguistic labels

For each training context, obtain:

- word segmentation;
- POS;
- NER;
- dependency relations;
- phrase/chunk boundaries;
- morphology/lexical classes where reliable;
- tone labels for tone-bearing tokens.

Tone should remain available because Wave 1 showed that it can be learned, but its role changes from **direct importance signal** to **auxiliary linguistic supervision**.

---

# 9. Proposed training objective

The current Wave 1 objective is:

```text
L_total = L_LM + 0.1 * L_tone
```

The proposed compressor should instead use:

\[
L = L_{rel}
  + \lambda_{ling}L_{ling}
  + \lambda_{dep}L_{dep}
  + \lambda_{faith}L_{faith}
  + \lambda_{budget}L_{budget}
\]

## 9.1 Relevance loss

\[
L_{rel}=BCE(y_i^{rel}, \hat y_i^{rel})
\]

This is the primary objective.

For severe class imbalance, evaluate focal loss or class-weighted BCE as an ablation.

## 9.2 Linguistic loss

\[
L_{ling}=L_{POS}+L_{NER}+L_{DEP}+L_{TONE}+L_{MORPH}
\]

Do not start with all heads simultaneously. Recommended staged rollout:

```text
Stage A: relevance only
Stage B: + POS + NER
Stage C: + dependency
Stage D: + morphology
Stage E: + tone
```

This makes the contribution attributable.

## 9.3 Structural support loss

Penalize cases where the model keeps a target/evidence node but removes a required support node.

For dependency/evidence edge \(i \rightarrow j\):

```text
KEEP(i) + DROP(j) where j is required support
        => penalty
```

This can be implemented initially as a pairwise consistency loss rather than a complicated graph neural network.

## 9.4 Budget loss

For target retention ratio \(r\):

\[
L_{budget}=\left|\frac{N_{keep}}{N}-r\right|
\]

Evaluate multiple target ratios such as 0.5, 0.25, 0.125, corresponding approximately to 2×, 4× and 8× compression.

## 9.5 Faithfulness loss

Optionally distill task behavior:

\[
L_{faith}=KL(P_{teacher}(Y|Q,C)\;||\;P_{student}(Y|Q,C'))
\]

where \(C'\) is the compressed context.

A cheaper first implementation can use answer consistency or teacher-generated answer equivalence rather than full token-level KL.

---

# 10. Hierarchical compression is preferable to global top-K

Recommended selection hierarchy:

```text
Document
   |
   v
Sentence relevance
   |
   v
Clause/phrase relevance
   |
   v
Token relevance
   |
   v
Dependency/evidence closure
   |
   v
Budget correction
   |
   v
Compressed context
```

This avoids pathological outputs such as:

```text
"Nguyễn Văn A ... giám đốc ... năm 1980"
```

when the actual evidence requires a coherent sentence.

A practical implementation does not need a new language model for each level. The same encoder can produce token scores; sentence/clause scores can be pooled from token scores, followed by structural restoration.

---

# 11. Dependency/evidence closure

This is the proposed E7-level contribution.

Construct an evidence graph from linguistic annotations and teacher evidence:

```text
Question
   |
   v
Entity X
   |
   +-- born_at --> Y
                     |
                     +-- located_in --> Z
```

If the selector retains Z because it is answer-bearing, but Z requires Y and the relation X→Y to remain interpretable, the compressor restores the required support.

Operationally:

```text
1. Rank sentences/tokens by relevance.
2. Select candidates under budget.
3. Build dependency/evidence graph.
4. Find required support for selected evidence.
5. Restore missing support.
6. If budget is exceeded, remove lower-value candidates.
7. Produce final compressed context.
```

This directly targets the referential-dangling failure mode identified in recent work.

---

# 12. Model selection strategy

## Primary compressor

**PhoBERT-base** is the recommended first model.

Reason:

- Vietnamese-specific pretraining;
- strong NLP representations;
- much smaller than a 4B generative scorer;
- naturally suited to token classification;
- compatible with POS/NER/dependency auxiliary tasks.

## Secondary models

- PhoBERT-large for scaling;
- XLM-R-base as a multilingual control;
- optionally a Vietnamese encoder/modern backbone if a strong 2026 candidate is validated.

## Critical experiment

```text
XLM-R-base
     vs
PhoBERT-base
```

Keep architecture, training data, teacher labels, budget and optimization identical.

If PhoBERT consistently performs better on Vietnamese compression and linguistic stress tests, the result supports the claim that Vietnamese-specific representation contributes to compression quality.

---

# 13. Training data strategy

Do not train the compressor directly on VCC-Bench test samples.

Current training corpus:

- `training_corpus_v1.json`;
- 149,693 Vietnamese Wikipedia paragraphs;
- deterministic 90/10 split;
- 134,723 train / 14,970 validation;
- approximately 27M tokens.

Current VCC-Bench v2 should remain a held-out evaluation benchmark.

## 13.1 Generate training QA from the corpus

Recommended pipeline:

```text
Vietnamese corpus
      |
      v
2K–8K token documents/chunks
      |
      v
Question generation
      |
      +--> direct lookup
      +--> multi-sentence
      +--> relational
      +--> multi-hop
      |
      v
Teacher evidence annotation
      |
      v
Vietnamese linguistic annotation
      |
      v
Training instances
```

## 13.2 Avoid train/test contamination

Before generating QA, remove any source documents overlapping with VCC-Bench evaluation documents or any known benchmark source.

Record:

- source document ID;
- dataset split;
- generation model/version;
- prompt version;
- linguistic parser version;
- teacher model/version;
- checksum.

---

# 14. Training question taxonomy

The training set should intentionally contain different evidence structures.

## Type A — Direct lookup

One sentence contains the answer.

## Type B — Multi-sentence

Answer requires combining 2–3 sentences.

## Type C — Relational

Question asks for relation between entities.

## Type D — Multi-hop

```text
A -> B -> C
```

The answer is impossible from one isolated span.

## Type E — Referential

Answer depends on antecedent, pronoun, abbreviation, or previously introduced entity.

## Type F — Distractor-heavy

Several plausible sentences contain related entities but only one evidence chain answers the query.

These categories are essential for determining whether linguistic/structural supervision genuinely helps rather than merely improving easy direct QA.

---

# 15. Benchmark redesign

Keep VCC-Bench v2 as the main held-out benchmark, but extend it with targeted diagnostics.

## 15.1 Existing tasks

- long-document QA;
- needle-in-haystack;
- summarization;
- cross-lingual;
- agent/tool-calling.

## 15.2 Recommended new diagnostic tasks

### B1 — Direct QA

One evidence sentence.

### B2 — Multi-sentence QA

Two or more evidence sentences.

### B3 — Multi-hop QA

Requires an evidence chain.

### B4 — Referential dependency

Answer depends on antecedent/bridge information.

### B5 — Vietnamese linguistic stress test

Include:

- compounds;
- reduplication;
- named entities;
- dates/numbers;
- Sino-Vietnamese terms;
- pronouns;
- ellipsis;
- ambiguous segmentation;
- punctuation-sensitive contexts;
- diacritic/decoy cases.

---

# 16. Metrics

Do not evaluate only with compression ratio and answer score.

## 16.1 Task performance

- Exact Match;
- Token-F1;
- ROUGE-L where applicable;
- BERTScore where applicable;
- tool-call correctness for agent tasks.

## 16.2 Compression

- compression ratio;
- retained-token ratio;
- latency;
- scorer memory/VRAM;
- total end-to-end cost.

## 16.3 Evidence Recall

\[
ER = \frac{|E \cap \hat{E}|}{|E|}
\]

where \(E\) is gold evidence and \(\hat E\) is retained evidence.

## 16.4 Evidence Closure

Suggested metric:

\[
EC = \frac{\text{retained evidence units with required support}}{\text{retained evidence units}}
\]

This measures whether the compressor keeps evidence in an interpretable form.

## 16.5 Linguistic Preservation

Measure retention of structures that are actually necessary for the task:

```text
POS preservation
NER preservation
Dependency preservation
Named-entity integrity
Tone/diacritic preservation
Phrase integrity
```

Do not optimize these blindly; report them as diagnostics tied to downstream performance.

## 16.6 Task Utility

A useful normalized metric:

\[
TU = \frac{Performance(compressed)}{Performance(full)}
\]

Plot utility against compression ratio rather than reporting one compression point only.

---

# 17. Required baseline matrix

At minimum:

```text
A0  No compression
A1  Random
A2  PPL
A3  Original LACC Wave 1
A4  Query-conditioned PPL / LongLLMLingua-style
A5  Query-only encoder compressor
A6  Query + POS/NER
A7  Query + POS/NER + DEP
A8  Query + linguistic + tone
A9  Query + linguistic + dependency closure (full model)
```

External baselines should include, where implementation permits:

- LLMLingua;
- LongLLMLingua;
- LLMLingua-2;
- QuerySelect / Adaptive QuerySelect;
- SnapKV where the evaluation protocol makes KV-cache comparison meaningful.

The comparison should be made at matched compression ratios and, where possible, matched latency/cost budgets.

---

# 18. Essential ablation study

The central question is whether linguistic supervision adds value **after** query conditioning.

Run:

```text
Query-only
      |
      +--> + POS
      |
      +--> + NER
      |
      +--> + DEP
      |
      +--> + POS + NER + DEP
      |
      +--> + morphology
      |
      +--> + tone
      |
      +--> + dependency closure
```

The most important comparisons are:

```text
Query-only
   vs
Query + linguistic
```

and:

```text
Query + linguistic
   vs
Query + linguistic + dependency closure
```

If these do not improve under matched conditions, the corresponding hypothesis should be rejected rather than rationalized.

---

# 19. The role of Tone Probe after Wave 1

Do not delete the Tone Probe from the research story.

Its new role is:

> **Controlled linguistic supervision experiment demonstrating that a model can learn a Vietnamese linguistic property without that property being a useful task-selection objective.**

The narrative becomes:

```text
Wave 1
Tone supervision
    |
    v
Learns tone successfully
    |
    v
But compression does not improve
    |
    v
Scientific lesson:
linguistic property != task relevance
    |
    v
Wave 2
Query relevance becomes primary target
    |
    v
Linguistic supervision becomes auxiliary
```

This is much stronger than simply hiding the failed hypothesis.

---

# 20. Wave 2 implementation roadmap

## Phase 1 — Query-conditioned baseline

**Goal:** prove that query conditioning fixes a major Wave 1 weakness.

Implement:

```text
Question + Context
        |
        v
PhoBERT-base
        |
        v
Relevance classifier
        |
        v
Keep/drop
```

Compare against PPL and LongLLMLingua/QuerySelect-style baselines.

**Go criterion:** clear improvement on QA/needle at matched compression ratio.

## Phase 2 — Vietnamese linguistic supervision

Add:

- POS;
- NER;
- dependency.

Run strict ablations.

**Go criterion:** query + linguistic > query-only on at least some controlled Vietnamese tasks, especially stress tests.

## Phase 3 — Structural completeness

Add dependency/evidence closure.

Target:

- multi-hop;
- referential;
- bridge-fact cases.

**Go criterion:** improved Evidence Closure and downstream performance under aggressive compression.

## Phase 4 — morphology and tone

Only after the main pipeline works, test:

- morphology;
- tone.

If they do not improve task utility, keep the negative/diagnostic result and do not force them into the final scoring formula.

---

# 21. Computational strategy for the current environment

The project does not need to repeat the expensive Qwen3-4B scorer approach for the main compressor.

Recommended priority:

```text
PhoBERT-base
   |
   +--> CPU smoke tests
   +--> single GPU training
   +--> full benchmark
```

Use larger models only for controlled scaling studies.

The main optimization target is:

> **task utility per unit compression cost**, not model size.

For query-conditioned PPL baselines, also test the proposed Wave 2 configuration change from 512/256 windows to approximately 2048/256 overlap, because the smaller sliding window causes redundant forward passes and distorts runtime comparisons.

---

# 22. E1/E2/E3 from the existing Wave 2 proposal should still be executed

## E1 — Query-conditioned/contrastive PPL

This is the lowest-risk experiment and should be run before training a new compressor.

```text
Context PPL
      vs
Question-conditioned PPL
```

Use a LongLLMLingua-style contrastive signal.

Expected strongest gain: needle and QA.

No model training is required.

## E2 — PPL × morphology

The existing morphology signal has some evidence of utility at zero extra VRAM, but additive blending can dilute PPL.

Test:

1. multiplicative morphology adjustment;
2. morphology as class-aware budget allocation.

Do not simply reuse the Wave 1 additive blend.

## E3 — Windowing/runtime correction

Change the scorer window configuration to reduce redundant computation before drawing cost conclusions.

Runtime fairness must be fixed before making claims about efficiency.

---

# 23. E4/E6/E7 should become the main training program

## E4 — Query relevance probe

Reuse the existing probe infrastructure but replace tone labels with relevance labels.

Current repo already contains:

```bash
python scripts/train_relevance_probe.py \
    --adapter-dir models/qwen3/final \
    --data-path data/benchmark/vcc_bench_v2.json \
    --output-dir models/qwen3 \
    --load-4bit
```

However, this should be treated as an **intermediate experiment**, not the final architecture.

Important correction: do not train relevance solely from exact answer tokens. Extend labels toward answer/evidence/support.

## E6 — Encoder token-classification compressor

The existing repo direction:

```bash
python scripts/train_encoder_compressor.py \
    --encoder-id vinai/phobert-base \
    --train-data-path data/benchmark/training_corpus_v1.json \
    --teacher-model Qwen/Qwen2.5-0.5B-Instruct \
    --ratio 4 \
    --output-dir models/encoder_cls
```

should become the main experimental backbone after E4 validates the relevance hypothesis.

## E7 — Dependency-aware selection

Implement:

```text
candidate selection
      |
      v
support graph
      |
      v
closure restoration
      |
      v
budget correction
```

This is the strongest proposed methodological extension.

---

# 24. Proposed paper contribution structure

A realistic paper should target four contributions.

## Contribution 1 — Empirical diagnosis

Show that linguistic-property supervision can be learned but does not necessarily improve task-aware compression.

Wave 1 provides this evidence.

## Contribution 2 — Query-conditioned Vietnamese compressor

A lightweight encoder learns explicit query relevance for Vietnamese context selection.

## Contribution 3 — Linguistic/structural supervision

Use Vietnamese linguistic annotations as auxiliary supervision and dependency/evidence constraints to preserve structurally necessary context.

## Contribution 4 — Vietnamese compression benchmark/diagnostics

VCC-Bench plus multi-hop, referential and linguistic stress tests.

The paper should not promise that every component is universally better. The experiments should identify **where** each component helps.

---

# 25. Recommended scientific hypotheses

## H1 — Query conditioning

> Query-conditioned compression significantly outperforms query-agnostic compression at matched compression ratios.

This is strongly motivated by LongLLMLingua and QuerySelect.

## H2 — Linguistic supervision

> Vietnamese linguistic auxiliary supervision improves relevance prediction and/or task utility over query-only supervision on linguistically challenging cases.

This is the main hypothesis that must be empirically tested.

## H3 — Structural completeness

> Dependency/evidence-aware restoration improves downstream performance under aggressive compression, particularly for multi-hop and referential tasks.

## H4 — Vietnamese specialization

> A Vietnamese-specific encoder provides better compression decisions than a multilingual encoder of comparable size under the same training protocol.

This can be tested directly with PhoBERT vs XLM-R.

---

# 26. Go / No-Go criteria

Pre-register these criteria internally before large-scale training.

## GO

Proceed with the full paper direction if:

```text
Query-only > PPL / query-agnostic baseline
```

and at least one of:

```text
Query + linguistic > Query-only
```

or:

```text
Query + linguistic + dependency
      > Query + linguistic
```

especially on multi-hop/referential Vietnamese cases.

## CONDITIONAL GO

If linguistic supervision does not improve aggregate performance but clearly improves a well-defined stress-test category, retain it as a targeted contribution.

## NO-GO FOR LINGUISTIC CLAIM

If:

```text
Query + linguistic ≈ Query-only
```

across all controlled experiments, do not claim that Vietnamese linguistic supervision improves compression. Reframe the work as a Vietnamese query-aware compressor/benchmark and report the negative result honestly.

## NO-GO FOR THE WHOLE APPROACH

If query conditioning itself fails to beat strong query-aware baselines under matched budgets and correct implementation, the research question should be reconsidered before investing in more auxiliary heads.

---

# 27. Recommended final research narrative

The strongest narrative is not:

> “Vietnamese has tones, therefore prompt compression should preserve tones.”

It is:

```text
Prompt compression is an information-selection problem.
             |
             v
PPL captures linguistic predictability,
not necessarily task relevance.
             |
             v
Wave 1 tests Vietnamese tone supervision.
             |
             v
Tone is learned but does not improve task compression.
             |
             v
Lesson: linguistic salience != task relevance.
             |
             v
Introduce query-conditioned relevance learning.
             |
             v
But relevance-only token selection can destroy
supporting/structural context.
             |
             v
Introduce Vietnamese linguistic supervision
and dependency/evidence completeness.
             |
             v
Evaluate under matched compression budgets,
including multi-hop/referential Vietnamese stress tests.
```

This narrative is scientifically cleaner because every stage follows from an observed limitation of the previous stage.

---

# 28. Recommended experiment table for the paper

| Model | Query | Linguistic supervision | Structural closure | Main purpose |
|---|---:|---:|---:|---|
| Full context | ✓ | — | — | Upper bound |
| Random | ✓ | — | — | Lower baseline |
| PPL | ✗ | ✗ | ✗ | Query-agnostic baseline |
| LLMLingua | ✗ | ✗ | ✗ | Strong established baseline |
| LongLLMLingua | ✓ | ✗ | ✗ | Query-aware baseline |
| QuerySelect | ✓ | ✗ | ✗ | Query-aware learned selection |
| LACC Wave 1 | ✗/late heuristic | tone/morph | ✗ | Previous method |
| Query-only | ✓ | ✗ | ✗ | New basic model |
| Query + linguistic | ✓ | POS/NER/DEP | ✗ | Main linguistic ablation |
| Query + linguistic + tone | ✓ | +tone | ✗ | Tone ablation |
| Full VQLC/LACC-2 | ✓ | ✓ | ✓ | Proposed model |

All should be compared at matched compression ratios such as 2×, 4× and 8×, with confidence intervals and paired bootstrap/significance testing where appropriate.

---

# 29. Reference list

## Context compression

1. Jiang et al. **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.** EMNLP 2023.  
   https://arxiv.org/abs/2310.05736

2. Jiang et al. **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.** ACL 2024.  
   https://aclanthology.org/2024.acl-long.91/

3. Pan et al. **LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.** Findings of ACL 2024.  
   https://aclanthology.org/2024.findings-acl.57/

4. **Fundamental Limits of Prompt Compression.** NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html

5. **PartPrompt: A Part-of-Speech Guided Prompt Compression Method.** 2024.  
   https://arxiv.org/abs/2409.15395

6. **Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors.** 2026.  
   https://arxiv.org/abs/2607.25335

7. **Referential Dangling in Context Compression.** 2026.  
   https://arxiv.org/abs/2608.04569

## Vietnamese NLP infrastructure

8. Nguyen & Nguyen. **PhoBERT: Pre-trained language models for Vietnamese.** Findings of EMNLP 2020.  
   https://aclanthology.org/2020.findings-emnlp.92/

9. Nguyen et al. **VnCoreNLP: A Vietnamese Natural Language Processing Toolkit.** NAACL 2018 Demonstrations.  
   https://aclanthology.org/N18-5012/

10. Phan et al. **ViT5: Pretrained Transformer-based Models for Vietnamese.** 2022.  
    https://arxiv.org/abs/2205.06457

11. Nguyen et al. **ViDeBERTa: A Powerful Pre-trained Language Model for Vietnamese.** EACL 2023.  
    https://github.com/HySonLab/ViDeBERTa

## Additional multilingual / Vietnamese references already tracked by VNCompress

The repository's `research/references.md` also tracks Vietnamese language models, tokenization, morphology, multilingual compression, KV-cache compression, latent context compression, and agent context management. That file should remain the project's broader bibliography; this document focuses on the references directly supporting the proposed research direction.

---

# 30. Mapping to the current repository

Current project components that should be retained:

```text
vncompress/compression.py
vncompress/linguistics.py
vncompress/training.py
vncompress/models.py
vncompress/evaluation.py
```

Existing Wave 2 entry points provide a useful starting point:

```text
scripts/train_relevance_probe.py
scripts/train_encoder_compressor.py
```

The following conceptual changes are recommended:

| Current component | Recommended future role |
|---|---|
| Tone heuristic | Diagnostic / linguistic baseline |
| Tone probe | Negative/control auxiliary task |
| PPL | Strong baseline and optional relevance feature |
| Morphology | Auxiliary feature / ablation |
| Query keyword multiplier | Replace with true query-conditioned representation |
| Global token top-K | Replace with hierarchical + structural selection |
| Qwen3-4B scorer | Secondary/scorer baseline, not default compressor |
| PhoBERT encoder | Primary learned compressor candidate |
| VCC-Bench v2 | Held-out main benchmark |
| VCC-Bench extension | Multi-hop/referential/linguistic diagnostics |

---

# 31. Immediate implementation order

### Step 0 — Correctness and reproducibility

Before new training:

- ensure the paper description matches actual code;
- remove/implement claims about iterative budget allocation and class-proportional budget;
- fix scorer windowing/runtime fairness;
- freeze dataset/version/checksums;
- ensure no benchmark leakage.

### Step 1 — Query-aware PPL

Implement and benchmark LongLLMLingua-style contrastive PPL.

### Step 2 — Query-only PhoBERT compressor

Build the minimum new learned baseline.

### Step 3 — Teacher evidence labels

Generate answer/evidence/support labels from training corpus only.

### Step 4 — Linguistic multi-task training

Add POS/NER/DEP one at a time.

### Step 5 — Structural closure

Implement evidence/dependency restoration.

### Step 6 — Tone/morphology ablations

Only retain them in the final model if the measurements justify their inclusion.

### Step 7 — Full benchmark and significance testing

Run all compression ratios and all model/generator robustness settings.

### Step 8 — Paper rewrite

Rewrite the abstract/introduction around the scientific progression rather than the original tone-first hypothesis.

---

# 32. Final recommendation

The project should now be treated as a **research program**, not merely a refactor of the original LACC scoring formula.

The strongest target is:

> **A lightweight Vietnamese query-conditioned compressor that learns task relevance, uses Vietnamese linguistic structure as auxiliary supervision, and preserves the structural/evidence dependencies needed for faithful downstream answers.**

The central scientific principle is:

```text
Linguistic salience
        !=
Task relevance
        !=
Evidence completeness
```

Wave 1 established the first distinction. Wave 2 should test whether explicitly modeling the second and third distinctions produces a better compressor.

The most important experimental chain is therefore:

```text
Wave 1 Tone
   |
   | learned, but ineffective for task selection
   v
E1 Query-conditioned PPL
   |
   v
E4 Query relevance
   |
   v
E6 Vietnamese linguistic supervision
   |
   v
E7 Dependency/evidence closure
   |
   v
VCC-Bench + multi-hop/referential/linguistic stress tests
```

If the data supports the hypotheses, this gives VNCompress a defensible and coherent paper story. If the data rejects one of the hypotheses, the document's ablation and go/no-go framework ensures that the project can pivot without forcing an unsupported claim.
