#!/usr/bin/env python3
"""
Tone-Aware Compression Training — HuggingFace Trainer + Custom Loss
====================================================================
Fine-tune LLMs with phonological consistency auxiliary loss using:
  - HuggingFace Trainer (with custom compute_loss)
  - PEFT/LoRA for parameter efficiency
  - VCC-Bench dataset as training data
  - Saves tone_probe weights for inference-time compression

Reference patterns:
  - HuggingFace Trainer subclassing: https://huggingface.co/docs/transformers/en/trainer_customize
  - Qwen LoRA fine-tuning: https://github.com/QwenLM/Qwen/blob/main/finetune.py
  - LLM Probing: https://github.com/gmeyoyan/LLM-Probing-Classifiers

Usage:
    python run_training.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
    python run_training.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --lambda-tone 0.1
    python run_training.py --model Qwen/Qwen2.5-1.5B-Instruct --use-qlora
"""
import argparse, os, sys, json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Dataset
# ============================================================================

class ToneTrainingDataset(Dataset):
    """Dataset that pairs text with per-token tone labels."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512,
                 tone_analyzer=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tone_analyzer is None:
            from vncompress.tone_aware import get_tone_analyzer, TONE_NAME_TO_ID
            tone_analyzer = get_tone_analyzer()
        self.tone_analyzer = tone_analyzer
        from vncompress.tone_aware import TONE_NAME_TO_ID
        self.TONE_NAME_TO_ID = TONE_NAME_TO_ID
        self.samples = []
        for text in texts:
            encoded = self._encode(text)
            if encoded and len(encoded['input_ids']) >= 10:
                self.samples.append(encoded)

    def _encode(self, text: str) -> Optional[Dict]:
        ids = self.tokenizer.encode(text, max_length=self.max_length,
                                     truncation=True, add_special_tokens=True)
        if len(ids) < 10:
            return None
        tone_labels = []
        for tid in ids:
            ts = self.tokenizer.decode([tid])
            ts = ts.replace('\u2581', ' ').replace('\u0120', ' ').strip()
            tn = self.tone_analyzer.get_dominant_tone(ts[:20])
            tone_labels.append(self.TONE_NAME_TO_ID.get(tn or 'ngang', 0))
        labels = ids.copy()
        return {'input_ids': ids, 'labels': labels, 'tone_labels': tone_labels,
                'length': len(ids)}

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


@dataclass
class ToneDataCollator:
    """Collates batches with padding. Tone labels match input shape."""
    pad_token_id: int
    max_length: int = 512

    def __call__(self, batch: List[Dict]) -> Dict:
        max_len = min(max(len(s['input_ids']) for s in batch), self.max_length)
        bs = len(batch)
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)
        tone_labels = torch.zeros(bs, max_len, dtype=torch.long)

        for i, s in enumerate(batch):
            L = min(len(s['input_ids']), max_len)
            input_ids[i, :L] = torch.tensor(s['input_ids'][:L])
            attn_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(s['labels'][:L])
            tl = s['tone_labels'][:L]
            tone_labels[i, :len(tl)] = torch.tensor(tl)

        return {'input_ids': input_ids, 'attention_mask': attn_mask,
                'labels': labels, 'tone_labels': tone_labels}


# ============================================================================
# Custom Trainer with Auxiliary Tone Loss
# ============================================================================

class ToneAwareTrainer:
    """
    Manual training loop with auxiliary phonological consistency loss.

    L_total = L_LM + lambda_tone * L_tone

    Uses raw PyTorch loop to avoid HuggingFace Trainer compatibility
    issues on some platforms.
    """
    def __init__(self, model, tokenizer, tone_criterion, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.tone_criterion = tone_criterion
        self.config = config
        self.device = next(model.parameters()).device

    def train(self, train_loader, eval_loader=None):
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        config = self.config
        model = self.model
        tone_criterion = self.tone_criterion

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get('lr', 2e-4), weight_decay=0.01
        )
        total_steps = len(train_loader) * config.get('epochs', 3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.get('warmup', 50),
            num_training_steps=total_steps
        )

        model.train()
        global_step = 0
        for epoch in range(config.get('epochs', 3)):
            print(f"\n--- Epoch {epoch+1}/{config.get('epochs',3)} ---")
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                tone_labels = batch['tone_labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attn,
                                labels=labels, output_hidden_states=True)
                lm_loss = outputs.loss
                hidden_states = outputs.hidden_states[-1]
                tone_loss = tone_criterion(hidden_states, tone_labels, attn)
                loss = lm_loss + tone_loss

                acc_steps = config.get('grad_accum', 4)
                loss = loss / acc_steps
                loss.backward()

                if (batch_idx + 1) % acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step > 0 and global_step % config.get('log_steps', 10) == 0:
                    print(f"  step={global_step} lm={lm_loss.item():.4f} "
                          f"tone={tone_loss.item():.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

                if config.get('eval_steps') and global_step % config['eval_steps'] == 0 and eval_loader:
                    self._eval(eval_loader)

                if config.get('max_steps', -1) > 0 and global_step >= config['max_steps']:
                    break

            if config.get('max_steps', -1) > 0 and global_step >= config['max_steps']:
                break

        return global_step

    def _eval(self, loader):
        self.model.eval()
        total_lm, total_tone, n = 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                tone_labels = batch['tone_labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attn,
                                     labels=labels, output_hidden_states=True)
                total_lm += outputs.loss.item()
                hs = outputs.hidden_states[-1]
                total_tone += self.tone_criterion(hs, tone_labels, attn).item()
                n += 1
        print(f"  [Eval] lm={total_lm/n:.4f} tone={total_tone/n:.4f}")
        self.model.train()


# ============================================================================
# Training Data
# ============================================================================

def load_training_texts(data_path: Optional[str] = None) -> List[str]:
    """Load training texts from VCC-Bench dataset or fallback to demo texts."""
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'paragraphs' in data:
            return [p['text'] for p in data['paragraphs'] if len(p['text']) > 200]
        if isinstance(data, dict) and 'samples' in data:
            return [s.get('context', '') for s in data['samples'] if len(s.get('context', '')) > 200]
        if isinstance(data, list):
            return [item if isinstance(item, str) else item.get('text', item.get('context', '')) for item in data]
        return []

    # Try VCC-Bench dataset (training_corpus_v1.json, if built via
    # scripts/build_training_corpus.py, takes priority -- it's a larger,
    # more diverse mix than the 393-paragraph Wikipedia-only snapshot).
    bench_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcc_bench_data', 'training_corpus_v1.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcc_bench_data', 'wikipedia_vi_raw.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcc_bench_data', 'vcc_bench_v1.json'),
    ]
    for bp in bench_paths:
        if os.path.exists(bp):
            return load_training_texts(bp)

    # Fallback: demo texts
    return _demo_texts()


def _demo_texts() -> List[str]:
    return [
        "Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
        "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
        "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. Bảo vệ môi trường "
        "là quyền, nghĩa vụ và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng dân cư, "
        "hộ gia đình và cá nhân." * 5,
        "Thị trường chứng khoán Việt Nam đã có phiên giao dịch tích cực vào ngày "
        "hôm nay khi chỉ số VN-Index tăng 12 điểm, đạt mức 1280 điểm. Khối lượng "
        "giao dịch đạt hơn 1 tỷ cổ phiếu với tổng giá trị giao dịch hơn 25 nghìn "
        "tỷ đồng. Nhóm cổ phiếu ngân hàng và bất động sản dẫn đầu đà tăng trưởng." * 5,
        "Trí tuệ nhân tạo đang phát triển nhanh chóng và có tác động sâu rộng đến "
        "mọi mặt của đời sống xã hội. Các mô hình ngôn ngữ lớn đã đạt được những "
        "tiến bộ vượt bậc trong việc hiểu và sinh văn bản tiếng Việt. Tuy nhiên, "
        "việc xử lý các văn bản tiếng Việt dài vẫn còn nhiều thách thức do đặc điểm "
        "ngôn ngữ đơn lập, có thanh điệu và nhiều từ ghép." * 5,
        "Học máy là một lĩnh vực của trí tuệ nhân tạo liên quan đến việc phát triển "
        "các thuật toán cho phép máy tính học từ dữ liệu. Có ba loại học máy chính: "
        "học có giám sát, học không giám sát và học tăng cường. Trong học có giám sát, "
        "mô hình được huấn luyện trên dữ liệu đã được gán nhãn." * 5,
    ]


# ============================================================================
# Main Training Function
# ============================================================================

def run_training(model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
                 output_dir: str = './trained_models',
                 num_epochs: int = 3, batch_size: int = 2,
                 learning_rate: float = 2e-4, max_length: int = 512,
                 lambda_tone: float = 0.1, lora_r: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05,
                 use_qlora: bool = False, max_steps: int = -1,
                 train_data_path: Optional[str] = None,
                 use_wandb: bool = False, device: str = 'cuda'):
    """Run tone-aware training with manual PyTorch loop."""

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from vncompress.tone_aware import PhonologicalConsistencyLoss

    print("=" * 60)
    print("VNCOMPRESS — Tone-Aware Training")
    print(f"Model: {model_name}  |  LoRA r={lora_r}  |  lambda_tone={lambda_tone}")
    print(f"Epochs: {num_epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print("=" * 60)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading model...")
    model_kwargs = dict(trust_remote_code=True,
                        torch_dtype=torch.float16)
    if device == 'cuda' and torch.cuda.is_available():
        model_kwargs['device_map'] = 'auto'
    if use_qlora:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.output_hidden_states = True

    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    print("[3/5] Preparing training data...")
    texts = load_training_texts(train_data_path)
    dataset = ToneTrainingDataset(texts, tokenizer, max_length=max_length)
    collator = ToneDataCollator(pad_token_id=tokenizer.pad_token_id, max_length=max_length)
    print(f"  {len(dataset)} training samples from {len(texts)} texts")

    split = int(0.9 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collator)
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    print("[4/5] Initializing tone loss...")
    hidden_dim = model.config.hidden_size
    tone_criterion = PhonologicalConsistencyLoss(hidden_dim=hidden_dim, lambda_tone=lambda_tone)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tone_criterion = tone_criterion.to(device=device, dtype=dtype)

    trainer = ToneAwareTrainer(
        model=model, tokenizer=tokenizer, tone_criterion=tone_criterion,
        config=dict(epochs=num_epochs, lr=learning_rate, warmup=50,
                    grad_accum=4, log_steps=10, eval_steps=None,
                    max_steps=max_steps),
    )

    print("[5/5] Training...")
    global_step = trainer.train(train_loader, eval_loader)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, 'final'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'final'))
    torch.save(tone_criterion.state_dict(), os.path.join(output_dir, 'tone_probe.pt'))
    print(f"\nSaved: {output_dir}/final + tone_probe.pt")

    # Quick eval
    model.eval()
    sample = dataset[0]
    ids = torch.tensor([sample['input_ids'][:max_length]]).to(trainer.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[-1]
    tl_tensor = torch.tensor([sample['tone_labels'][:max_length]]).to(hs.device)
    tc = tone_criterion.to(hs.device)
    logits = tc.tone_classifier(hs)
    preds = logits.argmax(dim=-1)
    correct = (preds == tl_tensor).float().mean().item()
    print(f"  Tone accuracy: {correct:.4f} ({correct*100:.1f}%)")
    print(f"  Steps: {global_step}")

    return model, tokenizer, tone_criterion


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train tone-aware compression model')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--output-dir', default='./trained_models')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--lambda-tone', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--use-qlora', action='store_true',
                        help='Use 4-bit QLoRA for limited VRAM')
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--train-data-path', default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 epoch, 1 batch, 50 steps')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    run_training(
        model_name=args.model, output_dir=args.output_dir,
        num_epochs=args.epochs if not args.quick else 1,
        batch_size=args.batch_size if not args.quick else 1,
        learning_rate=args.lr, max_length=args.max_length if not args.quick else 256,
        lambda_tone=args.lambda_tone, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora, max_steps=args.max_steps if not args.quick else 30,
        train_data_path=args.train_data_path, use_wandb=args.use_wandb,
        device=args.device,
    )


if __name__ == '__main__':
    main()
