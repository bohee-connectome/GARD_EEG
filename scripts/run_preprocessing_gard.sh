#!/bin/bash
# GARD EEG LMDB 변환 실행 스크립트
# Lab 서버 (node3)에서 실행
#
# 사용법:
#   ./run_preprocessing_gard.sh          # 전체 태스크 실행
#   ./run_preprocessing_gard.sh beam     # beam만 실행
#   ./run_preprocessing_gard.sh --debug  # 디버그 모드 (10개 파일만)

# 설정
DATA_PATH="/storage/bigdata/GARD/EEG/edf"
SAVE_PATH="/storage/bigdata/GARD/EEG/lmdb"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 가상환경 활성화 (필요시 수정)
# source /path/to/venv/bin/activate

# 인자 파싱
TASK="all"
DEBUG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        beam|sensory|attention|all)
            TASK="$1"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [beam|sensory|attention|all] [--debug]"
            exit 1
            ;;
    esac
done

# 실행 정보 출력
echo "=============================================="
echo "GARD EEG LMDB Conversion"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"
echo "Task: $TASK"
echo "Debug: ${DEBUG:-disabled}"
echo "=============================================="

# 디렉토리 확인
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path not found: $DATA_PATH"
    exit 1
fi

# 저장 디렉토리 생성
mkdir -p "$SAVE_PATH"

# 실행
echo ""
echo "Starting conversion..."
echo ""

python "$SCRIPT_DIR/preprocessing_gard.py" \
    --task "$TASK" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --resample_rate 500 \
    --segment_len 30 \
    --seed 42 \
    --eeg_only \
    $DEBUG

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Conversion completed!"
    echo "=============================================="
    echo ""
    echo "LMDB files saved to: $SAVE_PATH"
    echo ""

    # LMDB 검증
    if [ "$TASK" = "all" ]; then
        for task in beam sensory attention; do
            LMDB_FILE="$SAVE_PATH/$task/merged_resample-500_highpass-0.3_lowpass-200.lmdb"
            if [ -d "$LMDB_FILE" ]; then
                echo "Verifying $task..."
                python "$SCRIPT_DIR/check_lmdb_gard.py" "$LMDB_FILE"
            fi
        done
    else
        LMDB_FILE="$SAVE_PATH/$TASK/merged_resample-250_highpass-0.3_lowpass-200.lmdb"
        if [ -d "$LMDB_FILE" ]; then
            echo "Verifying $TASK..."
            python "$SCRIPT_DIR/check_lmdb_gard.py" "$LMDB_FILE"
        fi
    fi
else
    echo ""
    echo "=============================================="
    echo "Error: Conversion failed!"
    echo "=============================================="
    exit 1
fi
