#!/bin/bash

# ==============================================================================
# LLaMA Training Script - All Configurations
# ==============================================================================
# This script provides examples for running all optimizer and model combinations
# 
# USAGE:
#   ./RUN_TRAINING.sh
#
# Or run individual commands directly from the examples below
# ==============================================================================

echo "LLaMA Training - All Possible Configurations"
echo "=============================================="
echo ""
echo "This file contains all possible training commands."
echo "Uncomment the one you want to run, or copy-paste to terminal."
echo ""

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
GPU="2"                    # GPU device to use
BATCH_SIZE=32             # Batch size per GPU
MAX_STEPS=5000            # Total training steps
SUBSET_SIZE=""            # Leave empty for full dataset, or set like: --subset-size 10000

# ==============================================================================
# LLaMA-350M MODELS (4 optimizers)
# ==============================================================================

echo "# =========================================="
echo "# LLaMA-350M Models (Batch Size: $BATCH_SIZE)"
echo "# =========================================="
echo ""

# --- COSMOS on 350M ---
echo "## 1. COSMOS + LLaMA-350M"
echo "python3 train_llama_general.py \\"
echo "  --optimizer cosmos \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer cosmos --model-size 350m --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- SOAP on 350M ---
echo "## 2. SOAP + LLaMA-350M"
echo "python3 train_llama_general.py \\"
echo "  --optimizer soap \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer soap --model-size 350m --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- MUON on 350M ---
echo "## 3. MUON + LLaMA-350M"
echo "python3 train_llama_general.py \\"
echo "  --optimizer muon \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer muon --model-size 350m --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- AdamW on 350M ---
echo "## 4. AdamW + LLaMA-350M"
echo "python3 train_llama_general.py \\"
echo "  --optimizer adamw \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer adamw --model-size 350m --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# ==============================================================================
# LLaMA-1B MODELS (4 optimizers)
# ==============================================================================

echo "# =========================================="
echo "# LLaMA-1B Models (Batch Size: $BATCH_SIZE)"
echo "# =========================================="
echo ""

# --- COSMOS on 1B ---
echo "## 5. COSMOS + LLaMA-1B"
echo "python3 train_llama_general.py \\"
echo "  --optimizer cosmos \\"
echo "  --model-size 1b \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer cosmos --model-size 1b --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- SOAP on 1B ---
echo "## 6. SOAP + LLaMA-1B"
echo "python3 train_llama_general.py \\"
echo "  --optimizer soap \\"
echo "  --model-size 1b \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer soap --model-size 1b --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- MUON on 1B ---
echo "## 7. MUON + LLaMA-1B"
echo "python3 train_llama_general.py \\"
echo "  --optimizer muon \\"
echo "  --model-size 1b \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer muon --model-size 1b --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# --- AdamW on 1B ---
echo "## 8. AdamW + LLaMA-1B"
echo "python3 train_llama_general.py \\"
echo "  --optimizer adamw \\"
echo "  --model-size 1b \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS $SUBSET_SIZE"
echo ""

# Uncomment to run:
# python3 train_llama_general.py --optimizer adamw --model-size 1b --gpu $GPU --batch-size $BATCH_SIZE --max-steps $MAX_STEPS

# ==============================================================================
# CUSTOM LEARNING RATES
# ==============================================================================

echo "# =========================================="
echo "# Custom Learning Rates"
echo "# =========================================="
echo ""
echo "# Add --lr <value> to override default learning rate"
echo "# Example:"
echo "python3 train_llama_general.py \\"
echo "  --optimizer cosmos \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps $MAX_STEPS \\"
echo "  --lr 0.001"
echo ""

# ==============================================================================
# TESTING WITH SMALL SUBSET
# ==============================================================================

echo "# =========================================="
echo "# Quick Test with Data Subset"
echo "# =========================================="
echo ""
echo "# Use --subset-size for quick testing (smoke test)"
echo "# Example:"
echo "python3 train_llama_general.py \\"
echo "  --optimizer cosmos \\"
echo "  --model-size 350m \\"
echo "  --gpu $GPU \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --max-steps 100 \\"
echo "  --subset-size 1000"
echo ""

# ==============================================================================
# ALL AVAILABLE OPTIONS
# ==============================================================================

echo "# =========================================="
echo "# All Available Options"
echo "# =========================================="
echo ""
echo "python3 train_llama_general.py --help"
echo ""
echo "Arguments:"
echo "  --optimizer {cosmos,soap,muon,adamw}  Optimizer to use (default: cosmos)"
echo "  --model-size {350m,1b}                Model size (default: 350m)"
echo "  --gpu GPU                             GPU device (default: 2)"
echo "  --batch-size BATCH_SIZE               Batch size (default: 32)"
echo "  --max-steps MAX_STEPS                 Training steps (default: 5000)"
echo "  --subset-size SUBSET_SIZE             Data subset for testing (optional)"
echo "  --lr LR                               Custom learning rate (optional)"
echo ""

# ==============================================================================
# MODEL SPECIFICATIONS
# ==============================================================================

echo "# =========================================="
echo "# Model Specifications"
echo "# =========================================="
echo ""
echo "LLaMA-350M:"
echo "  Parameters: ~267M"
echo "  Hidden size: 1024"
echo "  Layers: 16"
echo "  Attention heads: 16"
echo "  FFN size: 2730"
echo "  Vocab size: 32000"
echo ""
echo "LLaMA-1B:"
echo "  Parameters: ~1.05B"
echo "  Hidden size: 2048"
echo "  Layers: 22"
echo "  Attention heads: 16"
echo "  FFN size: 5461"
echo "  Vocab size: 32000"
echo ""

# ==============================================================================
# OPTIMIZER DEFAULTS
# ==============================================================================

echo "# =========================================="
echo "# Optimizer Default Learning Rates"
echo "# =========================================="
echo ""
echo "COSMOS: lr=5e-4 (0.0005)"
echo "SOAP:   lr=3e-3 (0.003)"
echo "MUON:   lr=2e-2 (0.02)"
echo "AdamW:  lr=3e-4 (0.0003)"
echo ""

# ==============================================================================
# CHECKPOINTS & LOGS
# ==============================================================================

echo "# =========================================="
echo "# Outputs"
echo "# =========================================="
echo ""
echo "Checkpoints saved to: ./llama{model_size}_{optimizer}_checkpoints/"
echo "Logs directory: ./logs/"
echo "W&B logs: https://wandb.ai"
echo ""

# ==============================================================================
# EXAMPLE: START A TRAINING RUN
# ==============================================================================

echo "# =========================================="
echo "# To Start Training"
echo "# =========================================="
echo ""
echo "1. Choose a configuration from above"
echo "2. Copy the python command"
echo "3. Run it in your terminal"
echo ""
echo "Example to run COSMOS on 350M:"
echo ""
echo "  python3 train_llama_general.py --optimizer cosmos --model-size 350m --gpu 2 --batch-size 32 --max-steps 5000"
echo ""
echo "Example to run SOAP on 1B:"
echo ""
echo "  python3 train_llama_general.py --optimizer soap --model-size 1b --gpu 2 --batch-size 32 --max-steps 5000"
echo ""

# ==============================================================================
# MONITORING
# ==============================================================================

echo "# =========================================="
echo "# Monitoring Training"
echo "# =========================================="
echo ""
echo "Watch GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/train_*.log"
echo ""
echo "View on W&B:"
echo "  Check your W&B dashboard for real-time metrics"
echo ""
