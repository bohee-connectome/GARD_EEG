#!/bin/bash
###############################################################################
# GARD DIVER Integration Setup Script
#
# 이 스크립트를 Perlmutter에서 한 번만 실행하면 GARD finetuning 준비 완료
###############################################################################

set -e

DIVER_DIR="/global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "GARD DIVER Integration Setup"
echo "============================================================"
echo ""

# ============================================================
# Step 1: gard_dataset.py 복사
# ============================================================
echo "[1/3] Copying gard_dataset.py..."
cp "${SCRIPT_DIR}/gard_dataset.py" "${DIVER_DIR}/datasets/"
echo "      → ${DIVER_DIR}/datasets/gard_dataset.py"

# ============================================================
# Step 2: generalized_datasets.py 수정
# ============================================================
echo "[2/3] Patching generalized_datasets.py..."

GENERALIZED_FILE="${DIVER_DIR}/datasets/generalized_datasets.py"

# 이미 GARD가 추가되어 있는지 확인
if grep -q "from datasets.gard_dataset import GARDLoadDataset" "$GENERALIZED_FILE"; then
    echo "      → Already patched (skipping)"
else
    # import 추가 (다른 dataset import 뒤에)
    sed -i '/from datasets.embarc_dataset import/a from datasets.gard_dataset import GARDLoadDataset' "$GENERALIZED_FILE"

    # DATASETS_DICT에 GARD 추가
    sed -i "/'EMBARC'/a\\    'GARD': GARDLoadDataset," "$GENERALIZED_FILE"

    echo "      → Patched successfully"
fi

# ============================================================
# Step 3: finetune_main.py 수정
# ============================================================
echo "[3/3] Patching finetune_main.py..."

FINETUNE_FILE="${DIVER_DIR}/finetune_main.py"

# 이미 gard_task가 추가되어 있는지 확인
if grep -q "gard_task" "$FINETUNE_FILE"; then
    echo "      → Already patched (skipping)"
else
    # --gard_task argument 추가 (--embarc_preset 뒤에)
    sed -i "/--embarc_preset/a\\    parser.add_argument('--gard_task', type=str, default='task_a', choices=['task_a', 'task_b', 'task_c', 'task_d', 'task_e'], help='GARD task for progression label')" "$FINETUNE_FILE"

    echo "      → Patched successfully"
fi

# ============================================================
# Step 4: finetuning 스크립트 복사
# ============================================================
echo ""
echo "[+] Copying finetuning scripts..."
cp "${SCRIPT_DIR}"/run_finetune_GARD_*.sh "${DIVER_DIR}/"
echo "      → 6 scripts copied to ${DIVER_DIR}/"

# ============================================================
# Step 5: logs 디렉토리 생성
# ============================================================
mkdir -p "${DIVER_DIR}/logs"
echo "      → logs directory created"

# ============================================================
# 완료
# ============================================================
echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "이제 다음 명령어로 finetuning 실행:"
echo ""
echo "  cd ${DIVER_DIR}"
echo "  sbatch run_finetune_GARD_lora.sh"
echo "  sbatch run_finetune_GARD_dora.sh"
echo "  # ... 또는 전체 실행:"
echo "  for s in run_finetune_GARD_*.sh; do sbatch \$s; done"
echo ""
