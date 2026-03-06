#!/usr/bin/env python3
"""
LLaMA-350M Training with Multiple Optimizers
Supports: COSMOS, SOAP, MUON, AdamW

Paper: COSMOS - A Hybrid Adaptive Optimizer for Efficient Training of Large Language Models (ICLR 2026)
"""

import os
import sys
import argparse

# Parse GPU argument BEFORE any CUDA imports
parser = argparse.ArgumentParser(description="Train LLaMA with various optimizers")
parser.add_argument("--optimizer", type=str, default="cosmos",
                   choices=["cosmos", "soap", "muon", "adamw"],
                   help="Optimizer to use (default: cosmos)")
parser.add_argument("--model-size", type=str, default="350m",
                   choices=["350m", "1b"],
                   help="Model size to train (default: 350m)")
parser.add_argument("--gpu", type=str, default="2",
                   help="GPU device to use (default: 2)")
parser.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
parser.add_argument("--max-steps", type=int, default=5000,
                   help="Maximum training steps (default: 5000)")
parser.add_argument("--subset-size", type=int, default=None,
                   help="Use subset of data for testing (default: None = full dataset)")
parser.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: optimizer-specific)")
args = parser.parse_args()

# Set GPU BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Now safe to import torch and other libraries
import math
import logging
import platform
import subprocess
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import wandb

# Import custom optimizers
try:
    from cosmos_optimizer import COSMOS
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False

try:
    from soap_optimizer import SOAP
    SOAP_AVAILABLE = True
except ImportError:
    SOAP_AVAILABLE = False
    
try:
    from muon_optimizer import MUON
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training Hyperparameters
OPTIMIZER_NAME = args.optimizer.upper()
BATCH_SIZE = args.batch_size
MAX_STEPS = args.max_steps
SUBSET_SIZE = args.subset_size

# Optimizer-specific settings
if args.optimizer == "cosmos":
    if not COSMOS_AVAILABLE:
        print("❌ COSMOS optimizer not available. Make sure cosmos_optimizer.py exists.")
        sys.exit(1)
    COSMOS_LR = args.lr if args.lr else 5e-4
    ADAM_LR = 2e-3
    RANK = 64
    GAMMA = COSMOS_LR / ADAM_LR
elif args.optimizer == "soap":
    if not SOAP_AVAILABLE:
        print("❌ SOAP optimizer not available. Make sure soap_optimizer.py exists.")
        sys.exit(1)
    SOAP_LR = args.lr if args.lr else 3e-3
elif args.optimizer == "muon":
    if not MUON_AVAILABLE:
        print("❌ MUON optimizer not available. Make sure muon_optimizer.py exists.")
        sys.exit(1)
    MUON_LR = args.lr if args.lr else 2e-2
    ADAM_LR = 2e-3  # For embeddings
elif args.optimizer == "adamw":
    ADAMW_LR = args.lr if args.lr else 3e-4

# Common hyperparameters
WEIGHT_DECAY = 0.0
BETA1, BETA2 = 0.9, 0.98
WARMUP_STEPS = 500
SEQ_LEN = 1024
NUM_WORKERS = 4

# Checkpointing
SAVE_EVERY = 500
SAVE_DIR = f"./llama{args.model_size}_{args.optimizer}_checkpoints"
LOG_EVERY = 50

# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cosmos_train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("CUDA not available! Training will be very slow on CPU.")

# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_llama_350m() -> LlamaForCausalLM:
    """Build LLaMA-350M matching COSMOS paper specs (Appendix A.1)."""
    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=2730,       # ≈ (8/3) × 1024
        num_hidden_layers=16,
        num_attention_heads=16,       # head_dim = 64
        num_key_value_heads=16,       # Full MHA
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        vocab_size=32000,             # T5 tokenizer
        hidden_act="silu",
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"✅ Model built — {n_params:.1f}M parameters")
    logger.info(f"   Architecture: d={config.hidden_size}, L={config.num_hidden_layers}, "
                f"H={config.num_attention_heads}, FFN={config.intermediate_size}")
    return model


def build_llama_1b() -> LlamaForCausalLM:
    """Build LLaMA-1B with standard scaling.
    
    Follows LLaMA-2 scaling patterns:
    - Hidden size: 2048 (2x 350M)
    - Layers: 22 (scaling up from 16)
    - Heads: 16 (head_dim = 128)
    - FFN: ~5461 (≈ (8/3) × 2048)
    """
    config = LlamaConfig(
        hidden_size=2048,
        intermediate_size=5461,       # ≈ (8/3) × 2048
        num_hidden_layers=22,
        num_attention_heads=16,       # head_dim = 128
        num_key_value_heads=16,       # Full MHA
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        vocab_size=32000,             # T5 tokenizer
        hidden_act="silu",
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"✅ Model built — {n_params:.1f}M parameters")
    logger.info(f"   Architecture: d={config.hidden_size}, L={config.num_hidden_layers}, "
                f"H={config.num_attention_heads}, FFN={config.intermediate_size}")
    return model


def build_model(model_size: str) -> LlamaForCausalLM:
    """Build model based on size specification."""
    if model_size == "350m":
        return build_llama_350m()
    elif model_size == "1b":
        return build_llama_1b()
    else:
        raise ValueError(f"Unknown model size: {model_size}")


# ============================================================================
# OPTIMIZER BUILDER
# ============================================================================

def build_optimizer(model):
    """Build optimizer based on command-line argument."""
    
    if args.optimizer == "cosmos":
        logger.info(f"✅ Building COSMOS optimizer (lr={COSMOS_LR}, γ={GAMMA:.4f})")
        
        embed_params, cosmos_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                embed_params.append(param)
            else:
                cosmos_params.append(param)

        logger.info(
            f"COSMOS params: {sum(p.numel() for p in cosmos_params)/1e6:.1f}M  |  "
            f"AdamW params:  {sum(p.numel() for p in embed_params)/1e6:.1f}M"
        )

        cosmos_opt = COSMOS(
            cosmos_params,
            lr=COSMOS_LR,
            betas=(BETA1, BETA2),
            eps=1e-8,
            rank=RANK,
            gamma=GAMMA,
            weight_decay=WEIGHT_DECAY,
            nestrov=True,
            lr_ratio=0.1,
        )
        adam_opt = torch.optim.AdamW(
            embed_params,
            lr=ADAM_LR,
            betas=(BETA1, BETA2),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
        return [cosmos_opt, adam_opt]
    
    elif args.optimizer == "soap":
        logger.info(f"✅ Building SOAP optimizer (lr={SOAP_LR})")
        
        soap_opt = SOAP(
            model.parameters(),
            lr=SOAP_LR,
            betas=(BETA1, BETA2),
            weight_decay=WEIGHT_DECAY,
            precondition_frequency=10,
            max_precond_dim=10000,
        )
        return [soap_opt]
    
    elif args.optimizer == "muon":
        logger.info(f"✅ Building MUON optimizer (lr={MUON_LR})")
        
        # MUON for hidden layers, AdamW for embeddings
        embed_params, muon_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                embed_params.append(param)
            else:
                muon_params.append(param)

        logger.info(
            f"MUON params: {sum(p.numel() for p in muon_params)/1e6:.1f}M  |  "
            f"AdamW params: {sum(p.numel() for p in embed_params)/1e6:.1f}M"
        )

        muon_opt = MUON(
            muon_params,
            lr=MUON_LR,
            momentum=0.95,
            ns_steps=5,
        )
        adam_opt = torch.optim.AdamW(
            embed_params,
            lr=ADAM_LR,
            betas=(BETA1, BETA2),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
        return [muon_opt, adam_opt]
    
    elif args.optimizer == "adamw":
        logger.info(f"✅ Building AdamW optimizer (lr={ADAMW_LR})")
        
        adamw_opt = torch.optim.AdamW(
            model.parameters(),
            lr=ADAMW_LR,
            betas=(BETA1, BETA2),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY,
        )
        return [adamw_opt]


# ============================================================================
# DATALOADER
# ============================================================================

class TokenDataset(torch.utils.data.IterableDataset):
    """Wraps a HuggingFace streaming dataset, tokenizing on the fly."""
    def __init__(self, hf_dataset, tok, length):
        self.ds, self.tok, self.len = hf_dataset, tok, length

    def __iter__(self):
        for sample in self.ds:
            enc = self.tok(
                sample["text"],
                truncation=True,
                max_length=self.len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            yield {"input_ids": input_ids, "labels": input_ids.clone()}


def build_dataloader(tokenizer):
    """Build C4 dataloader."""
    raw_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    if SUBSET_SIZE is not None:
        raw_dataset = raw_dataset.take(SUBSET_SIZE)
        logger.info(f"Using subset of {SUBSET_SIZE} samples for smoke test")
    
    dataset = TokenDataset(raw_dataset, tokenizer, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS, pin_memory=True)
    logger.info(f"✅ Dataloader ready — C4 dataset, seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}")
    return loader


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function."""
    
    # Model architecture config based on model size
    if args.model_size == "350m":
        model_config = {
            "model": "LLaMA-350M",
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_size": 2730,
        }
    else:  # 1b
        model_config = {
            "model": "LLaMA-1B",
            "hidden_size": 2048,
            "num_layers": 22,
            "num_heads": 16,
            "intermediate_size": 5461,
        }
    
    # Initialize W&B
    logger.info("Initializing Weights & Biases...")
    wandb.init(
        project=f"llama-{args.model_size}-optimizers",
        name=f"llama{args.model_size}_{args.optimizer}_bs{BATCH_SIZE}",
        config={
            **model_config,
            "optimizer": OPTIMIZER_NAME,
            "vocab_size": 32000,
            "seq_len": SEQ_LEN,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE,
            "effective_tokens_per_step": BATCH_SIZE * SEQ_LEN,
            "dataset": "C4 (en)",
            "tokenizer": "T5-base",
            "gpu": args.gpu,
        }
    )
    
    # Add optimizer-specific config
    if args.optimizer == "cosmos":
        wandb.config.update({
            "cosmos_lr": COSMOS_LR,
            "adam_lr": ADAM_LR,
            "rank": RANK,
            "gamma": GAMMA,
        })
    elif args.optimizer == "soap":
        wandb.config.update({"soap_lr": SOAP_LR})
    elif args.optimizer == "muon":
        wandb.config.update({"muon_lr": MUON_LR, "adam_lr": ADAM_LR})
    elif args.optimizer == "adamw":
        wandb.config.update({"adamw_lr": ADAMW_LR})
    
    # Build model
    logger.info(f"Building LLaMA-{args.model_size.upper()} model...")
    model = build_model(args.model_size).to(device)
    
    # Build optimizers
    logger.info("Building optimizers...")
    optimizers = build_optimizer(model)
    
    # Build schedulers
    logger.info("Building schedulers...")
    schedulers = []
    for opt in optimizers:
        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=MAX_STEPS,
        )
        schedulers.append(sched)
    logger.info(f"✅ Schedulers ready — {MAX_STEPS} total steps, {WARMUP_STEPS} warmup steps")
    
    # Build dataloader
    logger.info("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.pad_token = tokenizer.eos_token
    loader = build_dataloader(tokenizer)
    
    # Setup checkpointing
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training loop
    scaler = GradScaler()
    global_step = 0
    
    logger.info("=" * 60)
    logger.info(f"Starting training — {OPTIMIZER_NAME} optimizer on LLaMA-350M")
    logger.info(f"Total steps: {MAX_STEPS} | Warmup: {WARMUP_STEPS}")
    logger.info(f"Batch size: {BATCH_SIZE} | Tokens/step: {BATCH_SIZE * SEQ_LEN}")
    logger.info("=" * 60)
    
    model.train()
    
    for batch in loader:
        if global_step >= MAX_STEPS:
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with automatic mixed precision
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer steps
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        
        # Zero gradients
        for opt in optimizers:
            opt.zero_grad()
        
        # Update learning rate schedulers
        for sched in schedulers:
            sched.step()
        
        global_step += 1
        
        # Log metrics to W&B
        log_dict = {
            "train/loss": loss.item(),
            "train/perplexity": math.exp(min(loss.item(), 20)),
            "step": global_step,
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }
        
        # Add learning rates
        for i, sched in enumerate(schedulers):
            log_dict[f"lr/optimizer_{i}"] = sched.get_last_lr()[0]
        
        wandb.log(log_dict)
        
        # Console logging
        if global_step % LOG_EVERY == 0:
            lr_str = " | ".join([f"LR{i}: {sched.get_last_lr()[0]:.2e}" for i, sched in enumerate(schedulers)])
            logger.info(
                f"Step {global_step}/{MAX_STEPS} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {math.exp(min(loss.item(), 20)):.2f} | "
                f"{lr_str}"
            )
        
        # Save checkpoints
        if global_step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_step_{global_step}.pt")
            ckpt_dict = {
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'loss': loss.item(),
            }
            for i, opt in enumerate(optimizers):
                ckpt_dict[f'optimizer_{i}_state_dict'] = opt.state_dict()
            for i, sched in enumerate(schedulers):
                ckpt_dict[f'scheduler_{i}_state_dict'] = sched.state_dict()
            
            torch.save(ckpt_dict, ckpt_path)
            logger.info(f"✅ Checkpoint saved → {ckpt_path}")
    
    # Final checkpoint
    final_ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_final_step_{global_step}.pt")
    ckpt_dict = {
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'loss': loss.item(),
    }
    for i, opt in enumerate(optimizers):
        ckpt_dict[f'optimizer_{i}_state_dict'] = opt.state_dict()
    
    torch.save(ckpt_dict, final_ckpt_path)
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Final checkpoint: {final_ckpt_path}")
    logger.info("=" * 60)
    
    wandb.finish()


if __name__ == "__main__":
    main()
