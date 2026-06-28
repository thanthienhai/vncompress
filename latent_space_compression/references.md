# Nguồn Tham Khảo — VNCompress Research Project

**Cập nhật:** 28/06/2026

Đây là danh sách đầy đủ các nguồn tham khảo được sử dụng trong toàn bộ dự án, kèm link truy xuất.

---

## A. PAPERS — Context Compression (Nén Ngữ Cảnh)

### Prompt Compression
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| A1 | Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of LLMs" | EMNLP 2023 | [arxiv:2310.05736](https://arxiv.org/abs/2310.05736) |
| A2 | Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression" | ACL 2024 | [arxiv:2310.06839](https://arxiv.org/abs/2310.06839) |
| A3 | Xu et al. "CORE: Less is More — Lightweight Prompt Compression for QA on Edge Devices" | 2026 | [arxiv:2606.20571](https://arxiv.org/abs/2606.20571) |
| A4 | Zheng et al. "DiffuMask: Diffusion Language Model for Token-level Prompt Pruning" | 2026 | [arxiv:2604.06627](https://arxiv.org/abs/2604.06627) |
| A5 | Xu et al. "K-Token Merging: Compressing Sequences in the Latent Embedding Space" | 2026 | [arxiv:2604.15153](https://arxiv.org/abs/2604.15153) |

### KV Cache Compression
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| B1 | Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation" | 2024 | [arxiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| B2 | Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs" | NeurIPS 2023 | [arxiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| B3 | Xiao et al. "Efficient Streaming Language Models with Attention Sinks" | ICLR 2024 | [arxiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| B4 | Ge et al. "FastGen: Model Tells You What to Discard — Adaptive KV Cache Compression" | ICLR 2024 | [arxiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| B5 | Lin et al. "CompressKV: Semantic-Retrieval-Guided KV-Cache Compression" | 2026 | [arxiv:2606.24467](https://arxiv.org/abs/2606.24467) |
| B6 | Kai et al. "InfoKV: Information-Aware KV Cache Compression for Long Reasoning" | 2026 | [arxiv:2606.26875](https://arxiv.org/abs/2606.26875) |
| B7 | Ni & Lao. "AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty" | 2026 | [arxiv:2606.17872](https://arxiv.org/abs/2606.17872) |
| B8 | Bhatnagar et al. "STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding" | 2026 | [arxiv:2606.08382](https://arxiv.org/abs/2606.08382) |

### Latent-Space Compression
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| C1 | Mu et al. "Learning to Compress Prompts with Gist Tokens" | NeurIPS 2023 | [arxiv:2304.08467](https://arxiv.org/abs/2304.08467) |
| C2 | Ge et al. "In-context Autoencoder for Context Compression in a LLM" | ICLR 2024 | [arxiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| C3 | Chevalier et al. "Adapting Language Models to Compress Contexts" (AutoCompressor) | EMNLP 2023 | [arxiv:2312.06674](https://arxiv.org/search/?query=Adapting+Language+Models+to+Compress+Contexts) |
| C4 | Tan et al. "LLoCO: Learning Long Contexts Offline" | 2024 | [arxiv:2404.07979](https://arxiv.org/abs/2404.07979) |
| C5 | Tang et al. "SeCo: Beyond Position Bias — Semantic Consistency Context Compression" | 2026 | [arxiv:2605.09463](https://arxiv.org/abs/2605.09463) |
| C6 | Liu et al. "MemoSight: Unifying Context Compression and Multi-Token Prediction" | 2026 | [arxiv:2604.14889](https://arxiv.org/abs/2604.14889) |
| C7 | Haghifam et al. "H2MT: Semantic Hierarchy-Aware Hierarchical Memory Transformer" | 2026 | [arxiv:2605.24930](https://arxiv.org/abs/2605.24930) |

### Agent Context Management
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| D1 | Mehta & Datta. "Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents" | 2026 | [arxiv:2606.22953](https://arxiv.org/abs/2606.22953) |
| D2 | Yang et al. "RaMem: Contextual Reinstatement for Long-term Agentic Memory" | 2026 | [arxiv:2606.22844](https://arxiv.org/abs/2606.22844) |
| D3 | Ghulyani et al. "PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents" | 2026 | [arxiv:2606.20047](https://arxiv.org/abs/2606.20047) |
| D4 | Trukhina & Vashkelis. "Context Codec: Compress the Context, Keep the Commitments" | 2026 | [arxiv:2605.17304](https://arxiv.org/abs/2605.17304) |
| D5 | Zhang & Sun. "AGORA: Adapter-Grounded Observation-Action Retention for LLM Agents" | 2026 | [arxiv:2605.26596](https://arxiv.org/abs/2605.26596) |
| D6 | Liu et al. "Relink: Safe to Check, Unsafe to Use — Relinking at Compression Boundary" | 2026 | [arxiv:2606.21732](https://arxiv.org/abs/2606.21732) |

### Multilingual & Low-Resource
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| E1 | Lee et al. "Equity with Efficiency: An Empirical Study of Tokenizers for Multilingual LLMs" | 2026 | [arxiv:2606.15044](https://arxiv.org/abs/2606.15044) |
| E2 | Deng et al. "The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference" | 2026 | [arxiv:2606.21869](https://arxiv.org/abs/2606.21869) |
| E3 | Colak. "Cross-Lingual Token Arbitrage: Optimizing Code Agent Context Windows via Local LLM Preprocessing" | Submitted EMNLP 2026 | [arxiv:2606.03618](https://arxiv.org/abs/2606.03618) |
| E4 | Guo et al. "Brain-LLM Alignment Tracks Training Data, Not Typology" | CoNLL 2026 | [arxiv:2605.23032](https://arxiv.org/abs/2605.23032) |
| E5 | Johnson. "The Compression Paradox: Provider-Dependent Energy Effects of Prompt Compression" | 2026 | [arxiv:2603.23528](https://arxiv.org/abs/2603.23528) |
| E6 | Johnson. "Compression Method Matters: Benchmark-Dependent Output Dynamics in LLM Prompt Compression" | 2026 | [arxiv:2603.23527](https://arxiv.org/abs/2603.23527) |

### Selective Context / Related
| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| F1 | Li et al. "Selective Context: Compressing Prompts for Efficient In-Context Learning" | 2023 | [arxiv:2310.06201](https://arxiv.org/abs/2310.06201) |
| F2 | Ge et al. "PyramidKV: Pyramid-shaped KV Cache Compression for Long-Context LLM Inference" | 2024 | [arxiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| F3 | Lee et al. "Dustin: Draft-Augmented Sparse Verification for Efficient Long-Context Generation" | ICML 2026 | [arxiv:2606.24957](https://arxiv.org/abs/2606.24957) |
| F4 | Liu et al. "ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation" | 2026 | [arxiv:2606.11164](https://arxiv.org/abs/2606.11164) |

---

## B. PAPERS — Vietnamese NLP & Language Models

| # | Paper | Venue | arXiv / Link |
|---|-------|-------|-------------|
| V1 | Nguyen & Nguyen. "PhoBERT: Pre-trained Language Models for Vietnamese" | EMNLP 2020 Findings | [arxiv:2003.00744](https://arxiv.org/abs/2003.00744) |
| V2 | Nguyen et al. "ViDeBERTa: A Powerful Pre-trained Language Model for Vietnamese" | EACL 2023 | [GitHub](https://github.com/HySonLab/ViDeBERTa) |
| V3 | Phan et al. "ViT5: Pretrained Transformer-based Models for Vietnamese" | 2022 | [arxiv:2205.06457](https://arxiv.org/abs/2205.06457) |
| V4 | Vo. "Vi-Mistral-X: Building a Vietnamese Language Model with Advanced Continual Pre-training" | 2024 | [arxiv:2403.15470](https://arxiv.org/abs/2403.15470) |
| V5 | Ta et al. "ViDia2Std: A Parallel Corpus and Methods for Vietnamese Dialect-to-Standard Translation" | AAAI 2026 Oral | [arxiv:2603.10211](https://arxiv.org/abs/2603.10211) |
| V6 | Nguyen et al. "DSC2025 ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs" | 2026 | [arxiv:2601.04711](https://arxiv.org/abs/2601.04711) |
| V7 | Huynh et al. "ViCLSR: Vietnamese Contrastive Learning for Sentence Representations" | 2026 | [arxiv:2603.21084](https://arxiv.org/abs/2603.21084) |
| V8 | Nguyen. "TextGraphFuseGAT: PhoBERT + Graph Attention for Vietnamese Token Classification" | VLSP 2025 | [arxiv:2510.11537](https://arxiv.org/abs/2510.11537) |
| V9 | Nguyen et al. "ViTextVQA: A Large-Scale VQA Dataset for Vietnamese Text in Images" | 2024 | [arxiv:2404.10652](https://arxiv.org/abs/2404.10652) |
| V10 | Sophia Maria. "Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia" (Shopee) | 2025 | [arxiv:2509.09121](https://arxiv.org/abs/2509.09121) |

---

## C. GITHUB REPOS — Vietnamese NLP Tools

| # | Repo | Link | Mô tả |
|---|------|------|-------|
| G1 | PhoBERT (VinAI) | [github.com/VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | Pre-trained BERT for Vietnamese |
| G2 | PhoNLP (VinAI) | [github.com/VinAIResearch/PhoNLP](https://github.com/VinAIResearch/PhoNLP) | Vietnamese NLP toolkit (POS, NER, dependency parsing) |
| G3 | RDRsegmenter | [github.com/datquocnguyen/RDRsegmenter](https://github.com/datquocnguyen/RDRsegmenter) | Fast Vietnamese word segmenter (LREC 2018) |
| G4 | underthesea | [github.com/undertheseanlp/underthesea](https://github.com/undertheseanlp/underthesea) | Vietnamese NLP toolkit |
| G5 | GPTViet | [github.com/VietnamAIHub/GPTViet](https://github.com/VietnamAIHub/GPTViet) | Bilingual Vietnamese-English foundation model |
| G6 | LaVy | [github.com/baochi0212/LaVy](https://github.com/baochi0212/LaVy) | Vietnamese Multimodal LLM |
| G7 | viBioGPT | [github.com/hungnlp/viBioGPT](https://github.com/hungnlp/viBioGPT) | Vietnamese medical LLM |
| G8 | VBD-LLaMA-3-8B | [HuggingFace](https://huggingface.co/VBD-LLaMA-3-8B) | LLaMA-3 fine-tuned for Vietnamese |
| G9 | Vietnamese Wordlist | [github.com/duyet/vietnamese-wordlist](https://github.com/duyet/vietnamese-wordlist) | 10K common Vietnamese words |
| G10 | Vietnam Sensitive Words | [github.com/behitek/vietnam-sensitive-words](https://github.com/behitek/vietnam-sensitive-words) | 5K sensitive + teencode Vietnamese words |
| G11 | Vietnamese WordNet | [github.com/zeloru/vietnamese-wordnet](https://github.com/zeloru/vietnamese-wordnet) | WordNet for Vietnamese |
| G12 | VietSentiWordNet | [github.com/sonvx/VietSentiWordNet](https://github.com/sonvx/VietSentiWordNet) | Vietnamese sentiment lexicon |
| G13 | SeaLLM | [github.com/DAMO-NLP-SG/SeaLLMs](https://github.com/DAMO-NLP-SG/SeaLLMs) | Multilingual LLM for Southeast Asian languages |

---

## D. BENCHMARKS & DATASETS

| # | Name | Link | Mô tả |
|---|------|------|-------|
| H1 | LongBench | [arxiv:2308.00632](https://arxiv.org/abs/2308.00632) / [github.com/THUDM/LongBench](https://github.com/THUDM/LongBench) | Long-context benchmark |
| H2 | RULER | [arxiv:2404.06654](https://arxiv.org/abs/2404.06654) | Needle-in-haystack benchmark |
| H3 | LongBench-v2 | [github.com/THUDM/LongBench](https://github.com/THUDM/LongBench) | Agentic long-context benchmark |
| H4 | SWE-bench Multilingual | [github.com/SWE-bench/SWE-bench-Multilingual](https://github.com/SWE-bench/SWE-bench-Multilingual) | Multilingual coding benchmark |
| H5 | mmPISA-bench | [arxiv:2606.07069](https://arxiv.org/abs/2606.07069) | 43-language reasoning benchmark |
| H6 | MIRACL | [github.com/project-miracl/miracl](https://github.com/project-miracl/miracl) | Multilingual retrieval benchmark |

---

## E. MODELS (Base Models)

| # | Model | Source | Link |
|---|-------|--------|------|
| M1 | Qwen-2.5-7B-Instruct | Alibaba | [huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| M2 | Llama-3.1-8B-Instruct | Meta | [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| M3 | Gemma-2-9B-IT | Google | [huggingface.co/google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) |
| M4 | SmolLM2-135M-Instruct | HuggingFace | [huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| M5 | SmolLM2-360M-Instruct | HuggingFace | [huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| M6 | Qwen2.5-0.5B-Instruct | Alibaba | [huggingface.co/Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

---

## F. SERVING / INFRASTRUCTURE

| # | Project | Link |
|---|---------|------|
| S1 | NVIDIA/kvpress | [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) |
| S2 | TurboQuant (Google, ICLR 2026) | [github.com/tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) |
| S3 | NVlabs/RocketKV (ICML 2025) | [github.com/NVlabs/RocketKV](https://github.com/NVlabs/RocketKV) |
| S4 | opengear-project/GEAR | [github.com/opengear-project/GEAR](https://github.com/opengear-project/GEAR) |
| S5 | microsoft/LLMLingua | [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) |
| S6 | vllm-project/vllm | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |

---

## G. KEY VIETNAMESE LINGUISTICS RESOURCES

| # | Resource | Link |
|---|----------|------|
| L1 | Vietnamese syllable database (~6,500 syllables) | Integrated in `vncompress/tone_aware/vietnamese_linguistics.py` |
| L2 | Vietnamese function word dictionary (~200 words) | Integrated in `vncompress/morphology/merge_policy.py` |
| L3 | Vietnamese reduplicative patterns (~80 pairs) | Integrated in `vncompress/morphology/merge_policy.py` |
| L4 | Vietnamese tone system (6 tones) | Nguyễn, Đình-Hoà. "Vietnamese." London Oriental and African Language Library, 1997 |
