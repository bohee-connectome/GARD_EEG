# GARD LoRA Finetuning 전체 워크플로우

## 개요

```
[Phase 1: 데이터 준비]     →  [Phase 2: DIVER 연동]  →  [Phase 3: 학습]

EDF → LMDB → 라벨 업데이트      DataLoader 복사         LoRA Finetuning
```

---

## Phase 1: 데이터 준비 (1회만 실행)

> ⚠️ 이미 완료된 경우 Phase 2로 건너뛰기

### Step 1.1: EDF → LMDB 변환

**스크립트**: `preprocessing_gard.py`

```bash
# Perlmutter에서 실행
cd /global/cfs/cdirs/m4727/DIVER/GARD_EEG/scripts

# EEG Task별 변환 (attention, beam, sensory 중 선택)
python preprocessing_gard.py \
    --eeg_task attention \
    --data_path /pscratch/sd/b/boheelee/GARD/EEG/edf \
    --save_path /pscratch/sd/b/boheelee/GARD/EEG/lmdb \
    --label_path /pscratch/sd/b/boheelee/GARD/labels \
    --eeg_only

# 또는 배치 스크립트
sbatch run_preprocessing_gard.sh
```

**출력**: `/pscratch/.../lmdb/attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb`

### Step 1.2: 라벨 업데이트 (선택)

**스크립트**: `update_labels_gard.py`

```bash
# 라벨 CSV가 업데이트된 경우에만 실행
python update_labels_gard.py \
    --lmdb_path /pscratch/.../lmdb/attention/merged_*.lmdb \
    --label_path /pscratch/.../labels
```

### Step 1.3: LMDB 검증

**스크립트**: `check_lmdb_gard.py`

```bash
python check_lmdb_gard.py \
    --lmdb_path /pscratch/.../lmdb/attention/merged_*.lmdb \
    --task task_a
```

**확인 사항**:
- [ ] train/val/test 분할 확인
- [ ] target task 라벨 분포 확인
- [ ] sample shape (2, 30, 500) 확인

---

## Phase 2: DIVER-Clinical 연동 (1회만 실행)

### Step 2.1: DataLoader 복사

```bash
# 로컬 → Perlmutter
scp /Users/default/GARD_EEG/scripts/gard_dataset.py \
    perlmutter:/global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src/datasets/
```

### Step 2.2: generalized_datasets.py 수정

```bash
# Perlmutter에서 수정
cd /global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src/datasets
vim generalized_datasets.py
```

**수정 내용**:

```python
# 1. Import 추가 (파일 상단)
from datasets.gard_dataset import GARDCustomDataset, GARDLoadDataset

# 2. datasets dict에 추가 (get_dataset_loader 함수 내)
datasets = {
    # ... 기존 데이터셋 ...
    'gard': (GARDLoadDataset, GARDCustomDataset),  # 추가
}
```

### Step 2.3: finetune_main.py 수정

```bash
cd /global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src
vim finetune_main.py
```

**수정 내용**:

```python
# get_parser_args() 함수에 추가
parser.add_argument('--gard_task', type=str, default='task_a',
                    choices=['task_a', 'task_b', 'task_c', 'task_d', 'task_e'],
                    help='GARD classification task')
```

### Step 2.4: 학습 스크립트 복사

```bash
# 로컬 → Perlmutter
mkdir -p perlmutter:/global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/scripts/gard
scp /Users/default/GARD_EEG/scripts/run_finetune_GARD_lora.sh \
    perlmutter:/global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/scripts/gard/
```

---

## Phase 3: LoRA Finetuning (매번 실행)

### Step 3.1: 단일 실행 (테스트)

```bash
# Perlmutter에서
cd /global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/scripts/gard

# Interactive 모드 (테스트용)
bash run_finetune_GARD_lora.sh
```

### Step 3.2: Multi-Seed 실행 (본 실험)

```bash
# 3개 시드로 실행
for seed in 42 123 456; do
    sbatch --export=SEED=$seed run_finetune_GARD_lora.sh
done
```

### Step 3.3: 다른 Task 실행

`run_finetune_GARD_lora.sh` 수정:

```bash
# Task 변경
GARD_TASK="task_c"  # task_a → task_c (Amyloid)
GARD_TASK_NAME="amyloid"
```

또는 환경변수로:

```bash
sbatch --export=SEED=42,GARD_TASK=task_c run_finetune_GARD_lora.sh
```

---

## 전체 스크립트 순서 요약

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 데이터 준비 (Perlmutter, 1회)                      │
├─────────────────────────────────────────────────────────────┤
│  1. preprocessing_gard.py      # EDF → LMDB                 │
│  2. update_labels_gard.py      # 라벨 업데이트 (선택)        │
│  3. check_lmdb_gard.py         # LMDB 검증                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: DIVER 연동 (Perlmutter, 1회)                      │
├─────────────────────────────────────────────────────────────┤
│  1. gard_dataset.py 복사       # datasets/ 폴더로           │
│  2. generalized_datasets.py 수정                            │
│  3. finetune_main.py 수정                                   │
│  4. run_finetune_GARD_lora.sh 복사                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: 학습 (Perlmutter, 매번)                           │
├─────────────────────────────────────────────────────────────┤
│  1. run_finetune_GARD_lora.sh  # LoRA finetuning            │
│     - Task A~E 선택                                         │
│     - Multi-seed (42, 123, 456)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 스크립트 역할 정리

| 스크립트 | 역할 | 실행 시점 |
|---------|------|----------|
| `preprocessing_gard.py` | EDF → LMDB 변환 | Phase 1 (1회) |
| `update_labels_gard.py` | LMDB 라벨 업데이트 | 라벨 변경 시 |
| `check_lmdb_gard.py` | LMDB 검증 | Phase 1 (1회) |
| `gard_dataset.py` | Custom DataLoader | DIVER 연동 (1회) |
| `run_finetune_GARD_lora.sh` | LoRA 학습 | Phase 3 (매번) |

---

## 현재 상태 체크리스트

### Phase 1: 데이터 준비
- [x] EDF → LMDB 변환 완료
- [x] 라벨 업데이트 완료 (task_a ~ task_e)
- [x] LMDB 검증 완료

### Phase 2: DIVER 연동
- [ ] gard_dataset.py 복사
- [ ] generalized_datasets.py 수정
- [ ] finetune_main.py 수정
- [ ] run_finetune_GARD_lora.sh 복사

### Phase 3: 학습
- [ ] Task A 테스트 실행
- [ ] Task A multi-seed (3 seeds)
- [ ] Task B~E 실행

---

## 문제 해결

### LMDB not found
```bash
# LMDB 경로 확인
ls -la /pscratch/sd/b/boheelee/GARD/EEG/lmdb/attention/
```

### ImportError: gard_dataset
```bash
# 파일 위치 확인
ls /global/cfs/cdirs/m4727/DIVER/DIVER-Clinical/src/datasets/gard_dataset.py
```

### KeyError: 'gard'
```bash
# generalized_datasets.py에서 'gard' 추가 확인
grep -n "gard" generalized_datasets.py
```

### No samples with target task
```bash
# 해당 task의 라벨 분포 확인
python check_lmdb_gard.py --lmdb_path ... --task task_a
```
