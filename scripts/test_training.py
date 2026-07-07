import os, sys
sys.path.insert(0, '.')
from vncompress.tone_aware import PhonologicalConsistencyLoss
from run_training import ToneTrainingDataset, ToneDataCollator, load_training_texts
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts = load_training_texts()
ds = ToneTrainingDataset(texts, tok, max_length=256)
print(f'Dataset: {len(ds)} samples')
sample = ds[0]
input_len = len(sample['input_ids'])
tone_len = len(sample['tone_labels'])
print(f'  Sample: {input_len} tokens, {tone_len} tone labels')
print(f'  Tone IDs (first 20): {sample["tone_labels"][:20]}')

collator = ToneDataCollator(pad_token_id=tok.pad_token_id, max_length=256)
batch = collator([ds[0], ds[1]])
print(f'  Batch: input_ids={list(batch["input_ids"].shape)}, tone_labels={list(batch["tone_labels"].shape)}')

pcl = PhonologicalConsistencyLoss(hidden_dim=896)
n_params = sum(p.numel() for p in pcl.parameters())
print(f'Tone criterion: {n_params} params')
print('All OK!')
