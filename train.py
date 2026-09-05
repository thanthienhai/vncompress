#!/usr/bin/env python3
"""train.py -- LACC model training and SLM/tone-probe training.

    python train.py --mode lacc --config configs/training.json
    python train.py --mode lacc --model Qwen/Qwen2.5-0.5B-Instruct --quick
    python train.py --mode slm  --train-data-path data/benchmark/training_corpus_v1.json
    python train.py --mode slm --validate --adapter-dir models/slm/final --tone-probe models/slm/tone_probe.pt

See docs/training.md for the full guide (dataset building, hardware tuning,
how to read validation numbers, and measuring the SLM's real effect on the
compression pipeline via benchmark.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vncompress import training
from vncompress.config import load_experiment_config, save_run_metadata, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['lacc', 'slm'], default='lacc',
                         help="'lacc': fine-tune a generation-sized model. 'slm': train the small "
                              "Vietnamese scorer model (requires CUDA).")
    parser.add_argument('--config', default=None, help='ExperimentConfig JSON (model/seed/device/output_dir)')
    parser.add_argument('--validate', action='store_true',
                         help='--mode slm only: validate a trained checkpoint instead of training')

    parser.add_argument('--model', default=None, help='Base model. Default depends on --mode.')
    parser.add_argument('--output-dir', default=None, help='Default: ./models/lacc or ./models/slm')
    parser.add_argument('--train-data-path', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda-tone', type=float, default=None)
    parser.add_argument('--lora-r', type=int, default=None)
    parser.add_argument('--grad-accum', type=int, default=None, help='--mode slm only')
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--use-qlora', action='store_true', help='--mode lacc only')
    parser.add_argument('--load-4bit', action='store_true', help='--mode slm only: 4-bit NF4 QLoRA base')
    parser.add_argument('--base-dtype', choices=['float32', 'bfloat16'], default='float32', help='--mode slm only')
    parser.add_argument('--no-gradient-checkpointing', action='store_true', help='--mode slm only')
    parser.add_argument('--quick', action='store_true', help='Fast smoke-test run (few steps)')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu', 'mps'], help='--mode lacc only (--mode slm requires CUDA)')

    # --validate only
    parser.add_argument('--adapter-dir', default=None, help='Default: ./models/slm/final')
    parser.add_argument('--tone-probe', default=None, help='Default: ./models/slm/tone_probe.pt')
    parser.add_argument('--no-adapter', action='store_true', help='Evaluate the raw base model (fair perplexity baseline)')
    parser.add_argument('--dump-per-sample', default=None, help='Write per-sample NLL for scripts/compare_slm_runs.py')

    args = parser.parse_args()

    if args.mode == 'slm' and args.validate:
        training.validate_slm(
            adapter_dir=args.adapter_dir or './models/slm/final',
            tone_probe_path=args.tone_probe or './models/slm/tone_probe.pt',
            train_data_path=args.train_data_path,
            max_length=args.max_length or 128,
            no_adapter=args.no_adapter,
            dump_per_sample=args.dump_per_sample,
            dtype=args.base_dtype,
            load_4bit=args.load_4bit,
        )
        return

    exp_config = load_experiment_config(
        config_path=args.config,
        cli_overrides={'model': args.model, 'output_dir': args.output_dir, 'device': args.device},
    )

    if args.mode == 'lacc':
        output_dir = args.output_dir or './models/lacc'
        os.makedirs(output_dir, exist_ok=True)
        save_run_metadata(output_dir, exp_config)
        set_seed(exp_config.seed)
        training.run_lacc_training(
            model_name=exp_config.model,
            output_dir=output_dir,
            num_epochs=1 if args.quick else (args.epochs or 3),
            batch_size=1 if args.quick else (args.batch_size or 2),
            learning_rate=args.lr or 2e-4,
            max_length=256 if args.quick else (args.max_length or 512),
            lambda_tone=0.1 if args.lambda_tone is None else args.lambda_tone,
            lora_r=args.lora_r or 16,
            use_qlora=args.use_qlora,
            max_steps=30 if args.quick else args.max_steps,
            train_data_path=args.train_data_path,
            device=exp_config.device,
        )
    else:
        # ExperimentConfig's default model (a 7B instruct model) doesn't fit
        # the SLM use case -- use it only if the caller explicitly passed
        # --model or --config with one; otherwise fall back to the tuned SLM default.
        model_name = args.model or 'chronopt-research/vietnamese-gpt2-base'
        output_dir = args.output_dir or './models/slm'
        set_seed(exp_config.seed)
        training.run_slm_training(
            model_name=model_name,
            output_dir=output_dir,
            train_data_path=args.train_data_path,
            epochs=1 if args.quick else (args.epochs or 3),
            batch_size=args.batch_size or 8,
            max_length=min(args.max_length or 256, 128) if args.quick else (args.max_length or 256),
            lr=args.lr or 1e-4,
            lora_r=args.lora_r or 8,
            lambda_tone=0.1 if args.lambda_tone is None else args.lambda_tone,
            grad_accum=args.grad_accum or 2,
            max_steps=30 if args.quick else args.max_steps,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            base_dtype=args.base_dtype,
            load_4bit=args.load_4bit,
        )


if __name__ == '__main__':
    main()
