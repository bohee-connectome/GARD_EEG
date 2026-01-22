#!/bin/bash
#SBATCH -A m4727
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -o logs/gard_tcl_lora_%j.out
#SBATCH -e logs/gard_tcl_lora_%j.err
#SBATCH -J GARD_TCL_LoRA

###############################################################################
# GARD TCL + LoRA Finetuning Script
###############################################################################

set -e

# ============================================================
# Environment Setup
# ============================================================
module load pytorch/2.0.1
source /global/homes/b/boheelee/miniconda3/etc/profile.d/conda.sh
conda activate diver

cd /global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src
mkdir -p logs

# ============================================================
# Configuration
# ============================================================
GARD_TASK="task_a"
GARD_TASK_NAME="progression_single"
EEG_TASK="attention"
LMDB_PATH="/pscratch/sd/b/boheelee/GARD/EEG/lmdb/${EEG_TASK}/merged_resample-500_highpass-0.3_lowpass-200.lmdb"

FOUNDATION_DIR="/global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/DIVER_Foundation_v1"
BACKBONE_CONFIG="DIVER_EEG_FINAL_model"
WIDTH=256
DEPTH=6

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_BIAS="none"
LORA_TARGET="CNN_patcher_500+CNN_patcher_50+encoder_all"

BATCH_SIZE=32
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-2

OUTPUT_PARENT_DIR="/pscratch/sd/b/boheelee/GARD/finetuning/${GARD_TASK_NAME}/tcl_lora"
mkdir -p $OUTPUT_PARENT_DIR

# ============================================================
# Run - Multi Seed
# ============================================================
echo "============================================================"
echo "GARD TCL + LoRA Finetuning"
echo "============================================================"

if [ ! -d "${LMDB_PATH}" ]; then
    echo "ERROR: LMDB not found: ${LMDB_PATH}"
    exit 1
fi

for seed in 41 42 43; do
    echo ""
    echo "=================================================="
    echo "Running [TCL + LoRA] for GARD with Seed $seed"
    echo "=================================================="

    MODEL_DIR="${OUTPUT_PARENT_DIR}/seed${seed}"
    mkdir -p $MODEL_DIR

    python finetune_main.py \
        --downstream_dataset GARD \
        --gard_task "${GARD_TASK}" \
        --datasets_dir "${LMDB_PATH}" \
        --model_dir "${MODEL_DIR}" \
        --num_of_classes 2 \
        --seed "${seed}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --optimizer AdamW \
        --clip_value 1.0 \
        --dropout 0.1 \
        --backbone_config "${BACKBONE_CONFIG}" \
        --foundation_dir "${FOUNDATION_DIR}" \
        --width "${WIDTH}" \
        --depth "${DEPTH}" \
        --mup_weights True \
        --deepspeed_pth_format True \
        --frozen False \
        --feature_extraction_type multi_head_take_org_x \
        --ft_config flatten_linear \
        --ft_mup False \
        --use_lora True \
        --lora_type vanilla \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --lora_bias "${LORA_BIAS}" \
        --lora_target_name_group "${LORA_TARGET}" \
        --use_tcl True \
        --tcl_lambda 0.1 \
        --tcl_warmup_epochs 30 \
        --projection_dim 128 \
        --tcl_temp 0.07 \
        --tcl_num_subjects 8 \
        --tcl_k1 1.0 \
        --tcl_k2 1.5 \
        --early_stop_criteria val_f1 \
        --early_stop_patience 20 \
        --use_amp False \
        --use_dp False

    echo "Finished Seed $seed"
done

echo ""
echo "Generating Summary Report..."
python summarize_results.py --output_dir "$OUTPUT_PARENT_DIR"

echo "✅ All TCL + LoRA Finetuning Complete! → ${OUTPUT_PARENT_DIR}"
