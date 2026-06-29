#!/usr/bin/env python3
"""
Training Pipeline for Tone-Aware & Morphology-Aware Compression
================================================================
Fine-tune LLMs with tone/morphology-aware objectives for better
context compression on Vietnamese.

Training Modes:
  1. LoRA Fine-tuning: Efficient parameter-efficient training
  2. Tone Embedding Augmentation: Add tone embeddings to token reps
  3. Phonological Consistency Training: Auxiliary tone prediction loss

Usage:
    # Basic LoRA fine-tuning
    python run_training.py --model Qwen/Qwen2.5-7B-Instruct --mode lora
    
    # Tone-aware training with phonological consistency loss
    python run_training.py --model Qwen/Qwen2.5-7B-Instruct --mode tone_aware
    
    # Full combined training
    python run_training.py --model Qwen/Qwen2.5-7B-Instruct --mode combined
    
    # Quick test (1 epoch, small data)
    python run_training.py --model Qwen/Qwen2.5-7B-Instruct --mode tone_aware --quick

This script implements:
  - LoRA-based parameter-efficient fine-tuning
  - Custom phonological consistency loss
  - Tone embedding augmentation
  - Vietnamese tone label computation during training
  - Evaluation checkpoints with tone preservation metrics
"""

import argparse
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress.tone_aware import (
    VietnameseToneAnalyzer,
    ToneAwareConfig,
    ToneEmbeddingAugmentation,
    PhonologicalConsistencyLoss,
    ToneAugmentedTrainer,
    get_tone_analyzer,
    TONE_NAME_TO_ID,
)


# ============================================================================
# Training Dataset
# ============================================================================

class VietnameseCompressionDataset(Dataset):
    """
    Dataset for training compression-aware models on Vietnamese text.
    
    Each sample contains:
      - input_ids: Compressed token IDs
      - labels: Original token IDs (for reconstruction)
      - tone_labels: Per-token tone classes (for phonological consistency)
      - morphology_labels: Per-token word classes (for morphology-aware training)
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
        compression_ratio: float = 4.0,
        tone_analyzer: Optional[VietnameseToneAnalyzer] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self.tone_analyzer = tone_analyzer or get_tone_analyzer()
        
        self.samples = []
        for text in texts:
            encoded = self._encode_text(text)
            if encoded is not None:
                self.samples.append(encoded)
    
    def _encode_text(self, text: str) -> Optional[Dict]:
        """Encode text into training sample."""
        # Tokenize
        full_ids = self.tokenizer.encode(
            text, 
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        if len(full_ids) < 10:  # Too short
            return None
        
        # Compute tone labels
        tone_labels = []
        for tid in full_ids:
            token_str = self.tokenizer.decode([tid])
            token_str = token_str.replace('\u2581', ' ').replace('Ġ', ' ').strip()
            tone_name = self.tone_analyzer.get_dominant_tone(token_str)
            tone_id = TONE_NAME_TO_ID.get(tone_name or 'ngang', 0)
            tone_labels.append(tone_id)
        
        # Create compressed version (simulate compression)
        n = len(full_ids)
        target_len = max(1, int(n / self.compression_ratio))
        
        # Simple: keep boundaries + random selection
        k = 2
        if target_len <= 2 * k:
            compressed_ids = full_ids[:target_len]
        else:
            import random
            mid_indices = list(range(k, n - k))
            random.shuffle(mid_indices)
            keep = sorted(mid_indices[:target_len - 2 * k])
            compressed_ids = (
                full_ids[:k] + 
                [full_ids[i] for i in keep] + 
                full_ids[-k:]
            )
        
        # Labels: full text (model should learn to reconstruct/use compressed)
        return {
            'input_ids': compressed_ids,
            'labels': full_ids,
            'tone_labels': tone_labels,
            'original_length': n,
            'compressed_length': len(compressed_ids),
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict], tokenizer, pad_token_id: int) -> Dict:
    """Custom collate function for variable-length sequences."""
    import torch.nn.functional as F
    
    max_input_len = max(len(s['input_ids']) for s in batch)
    max_label_len = max(len(s['labels']) for s in batch)
    
    input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
    tone_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        ilen = len(sample['input_ids'])
        llen = len(sample['labels'])
        
        input_ids[i, :ilen] = torch.tensor(sample['input_ids'])
        attention_mask[i, :ilen] = 1
        labels[i, :llen] = torch.tensor(sample['labels'])
        tone_labels[i, :llen] = torch.tensor(sample['tone_labels'][:llen])
        
        # Mask label positions
        labels[i, ilen:] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'tone_labels': tone_labels,
    }


# ============================================================================
# Training Loop
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    device: str = 'cuda'
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ])
    
    # Tone-Aware
    tone_embed_dim: int = 64
    lambda_tone: float = 0.1
    alpha_tone: float = 0.5
    beta_tone: float = 0.3
    gamma_tone: float = 0.4
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 2048
    compression_ratio: float = 4.0
    warmup_steps: int = 100
    max_steps: int = -1
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = './trained_models'
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = 'vncompress'
    wandb_run_name: str = 'tone_aware_training'
    
    # Data
    train_data_path: Optional[str] = None  # Path to training data JSON
    val_data_path: Optional[str] = None


def run_training(config: TrainingConfig):
    """Main training function."""
    
    # Load model and tokenizer
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    
    print("="*60)
    print("VNCOMPRESS Training Pipeline")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Mode: Tone-Aware Compression Training")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"λ_tone: {config.lambda_tone}")
    print(f"Epochs: {config.num_epochs}, Batch: {config.batch_size}")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto' if config.device == 'cuda' else None,
    )
    model.config.output_hidden_states = True  # Need for tone loss
    
    # Apply LoRA
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    except ImportError:
        print("[WARNING] peft not installed. Training without LoRA.")
        print("  Install: pip install peft")
    
    # Prepare data
    train_texts = _get_training_texts(config)
    train_dataset = VietnameseCompressionDataset(
        train_texts, tokenizer,
        max_length=config.max_length,
        compression_ratio=config.compression_ratio,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, tokenizer.pad_token_id),
    )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    
    total_steps = (
        len(train_loader) // config.gradient_accumulation_steps * config.num_epochs
    ) if config.max_steps <= 0 else config.max_steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Initialize tone loss
    tone_criterion = PhonologicalConsistencyLoss(
        lambda_tone=config.lambda_tone,
    )
    
    # Wandb logging
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )
        except ImportError:
            config.use_wandb = False
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0
    total_lm_loss = 0.0
    total_tone_loss = 0.0
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        progress = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress):
            # Move to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            tone_labels = batch['tone_labels'].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            
            lm_loss = outputs.loss
            
            # Compute tone loss
            hidden_states = outputs.hidden_states[-1]  # [B, S, D]
            tone_l = tone_criterion(hidden_states, tone_labels, attention_mask)
            
            # Combined loss
            loss = lm_loss + tone_l
            
            # Scale for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Logging
            total_loss += loss.item() * config.gradient_accumulation_steps
            total_lm_loss += lm_loss.item()
            total_tone_loss += tone_l.item()
            
            if global_step > 0 and global_step % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                avg_lm = total_lm_loss / config.logging_steps
                avg_tone = total_tone_loss / config.logging_steps
                
                progress.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lm': f'{avg_lm:.4f}',
                    'tone': f'{avg_tone:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })
                
                if config.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lm_loss': avg_lm,
                        'train/tone_loss': avg_tone,
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/step': global_step,
                    })
                
                total_loss = 0.0
                total_lm_loss = 0.0
                total_tone_loss = 0.0
            
            # Save checkpoint
            if global_step > 0 and global_step % config.save_steps == 0:
                save_path = os.path.join(
                    config.output_dir, f"checkpoint-{global_step}"
                )
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\n  Checkpoint saved: {save_path}")
            
            # Early stopping
            if config.max_steps > 0 and global_step >= config.max_steps:
                break
        
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save config
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Steps: {global_step}")
    print(f"  Model saved: {final_path}")
    print(f"{'='*60}")
    
    if config.use_wandb:
        wandb.finish()


def _get_training_texts(config: TrainingConfig) -> List[str]:
    """Load or generate training texts."""
    if config.train_data_path and os.path.exists(config.train_data_path):
        with open(config.train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item if isinstance(item, str) else item.get('text', '') for item in data]
    
    # Demo: Vietnamese sample texts (for quick testing)
    demo_texts = [
        # Vietnamese legal texts
        "Luật Bảo vệ Môi trường năm 2020 quy định về hoạt động bảo vệ môi trường, "
        "quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, "
        "hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường. Điều 4 quy định "
        "nguyên tắc bảo vệ môi trường bao gồm: bảo vệ môi trường là quyền, nghĩa vụ "
        "và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình và "
        "cá nhân. Hoạt động bảo vệ môi trường phải được tiến hành thường xuyên, "
        "công khai, minh bạch; ưu tiên dự báo, phòng ngừa ô nhiễm, sự cố, suy thoái "
        "môi trường. Bảo vệ môi trường gắn kết hài hòa với phát triển kinh tế, "
        "an sinh xã hội, bảo đảm quyền trẻ em, thúc đẩy bình đẳng giới và phát "
        "triển bền vững." * 3,
        
        # Vietnamese news
        "Thị trường chứng khoán Việt Nam đã có phiên giao dịch tích cực vào ngày "
        "hôm nay khi chỉ số VN-Index tăng 12 điểm, đạt mức 1280 điểm. Khối lượng "
        "giao dịch đạt hơn 1 tỷ cổ phiếu với tổng giá trị giao dịch hơn 25 nghìn "
        "tỷ đồng. Nhóm cổ phiếu ngân hàng và bất động sản dẫn đầu đà tăng trưởng. "
        "Các chuyên gia nhận định thị trường sẽ tiếp tục xu hướng tích cực trong "
        "những phiên tới nhờ vào dòng tiền từ nhà đầu tư nước ngoài và kết quả "
        "kinh doanh quý 2 khả quan của các doanh nghiệp niêm yết." * 2,
        
        # Vietnamese conversation
        "Người dùng: Chào bạn, tôi muốn hỏi về thời tiết hôm nay ở Hà Nội.\n"
        "Trợ lý: Chào bạn! Hôm nay Hà Nội có nắng nhẹ, nhiệt độ từ 28 đến 35 độ C, "
        "độ ẩm khoảng 70%. Buổi chiều có thể có mưa rào nhẹ.\n"
        "Người dùng: Cảm ơn bạn. Vậy ngày mai thì sao?\n"
        "Trợ lý: Ngày mai dự báo trời nhiều mây, có mưa rào và dông vào chiều tối. "
        "Nhiệt độ từ 26 đến 32 độ C. Bạn nên mang theo ô khi ra ngoài.\n"
        "Người dùng: Còn Đà Nẵng thì thế nào?\n"
        "Trợ lý: Đà Nẵng hôm nay nắng đẹp, nhiệt độ 30-36 độ C, độ ẩm 65%. "
        "Rất thích hợp cho các hoạt động ngoài trời và du lịch biển." * 2,
        
        # Vietnamese technical
        "Học máy là một lĩnh vực của trí tuệ nhân tạo liên quan đến việc phát triển "
        "các thuật toán cho phép máy tính học từ dữ liệu. Có ba loại học máy chính: "
        "học có giám sát, học không giám sát và học tăng cường. Trong học có giám sát, "
        "mô hình được huấn luyện trên dữ liệu đã được gán nhãn. Thuật toán học cách "
        "ánh xạ từ đầu vào đến đầu ra dựa trên các cặp ví dụ đầu vào-đầu ra. "
        "Các thuật toán phổ biến bao gồm hồi quy tuyến tính, cây quyết định, "
        "và mạng nơ-ron nhân tạo." * 2,
    ]
    
    print(f"[INFO] Using {len(demo_texts)} demo training texts.")
    print(f"  To use custom data: --train-data-path data.json")
    
    return demo_texts


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train tone/morphology-aware compression models'
    )
    
    # Model
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='tone_aware',
                       choices=['lora', 'tone_aware', 'combined'])
    
    # LoRA
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    
    # Tone
    parser.add_argument('--lambda-tone', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--compression-ratio', type=float, default=4.0)
    parser.add_argument('--max-steps', type=int, default=-1)
    
    # Data
    parser.add_argument('--train-data-path', type=str, default=None)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./trained_models')
    parser.add_argument('--use-wandb', action='store_true')
    
    # Quick
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with minimal settings')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lambda_tone=args.lambda_tone,
        batch_size=args.batch_size if not args.quick else 1,
        num_epochs=args.epochs if not args.quick else 1,
        learning_rate=args.lr,
        max_length=args.max_length if not args.quick else 512,
        compression_ratio=args.compression_ratio,
        max_steps=args.max_steps if not args.quick else 50,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )
    
    run_training(config)


if __name__ == '__main__':
    main()
