# GARD DataLoader Integration Guide

## 개요

DIVER-Clinical의 `generalized_datasets.py`에 GARD 데이터셋을 연동하는 방법.

---

## 파일 구조

```
DIVER-Clinical/
├── src/
│   ├── datasets/
│   │   ├── generalized_datasets.py   # 수정 필요
│   │   ├── gard_dataset.py           # 복사 필요 (신규)
│   │   └── ...
│   └── finetune_main.py              # 수정 필요
└── scripts/
    └── gard/
        └── run_finetune_GARD_lora.sh # 복사 필요 (신규)
```

---

## Step 1: gard_dataset.py 복사

```bash
# GARD_EEG 레포에서 DIVER-Clinical로 복사
cp /path/to/GARD_EEG/scripts/gard_dataset.py \
   /path/to/DIVER-Clinical/src/datasets/gard_dataset.py
```

---

## Step 2: generalized_datasets.py 수정

### 2.1 Import 추가

```python
# 파일 상단에 추가
from datasets.gard_dataset import GARDCustomDataset, GARDLoadDataset
```

### 2.2 datasets dict에 추가

`get_dataset_loader()` 함수의 `datasets` 딕셔너리에 추가:

```python
def get_dataset_loader(dataset_name, params):
    dataset_name = dataset_name.lower()

    datasets = {
        'seedv': (LMDBLoadDataset, SEEDVCustomDataset),
        'faced': (LMDBLoadDataset, FACEDCustomDataset),
        # ... 기존 데이터셋들 ...

        # GARD 추가
        'gard': (GARDLoadDataset, GARDCustomDataset),
    }

    # ... 나머지 코드 ...
```

---

## Step 3: finetune_main.py 수정

### 3.1 Argument 추가

```python
# GARD specific argument
parser.add_argument('--gard_task', type=str, default='task_a',
                    choices=['task_a', 'task_b', 'task_c', 'task_d', 'task_e'],
                    help='GARD classification task')
```

### 3.2 TASK_DICT 추가 (선택사항)

`generalized_finetuning_utils/task_attribute_lookup.py`에 추가:

```python
FINAL_TASK_DICT = {
    # ... 기존 태스크들 ...

    # GARD Tasks
    'GARD_task_a': {
        'num_classes': 2,
        'task_type': 'classification',
        'name': 'Progression Single'
    },
    'GARD_task_b': {
        'num_classes': 2,
        'task_type': 'classification',
        'name': 'Progression Paired'
    },
    'GARD_task_c': {
        'num_classes': 2,
        'task_type': 'classification',
        'name': 'Amyloid Prediction'
    },
    'GARD_task_d': {
        'num_classes': 2,
        'task_type': 'classification',
        'name': 'Hippocampus Atrophy'
    },
    'GARD_task_e': {
        'num_classes': 2,
        'task_type': 'classification',
        'name': 'MTL Atrophy'
    },
}
```

---

## Step 4: 학습 스크립트 복사

```bash
# GARD 학습 스크립트 복사
mkdir -p /path/to/DIVER-Clinical/scripts/gard
cp /path/to/GARD_EEG/scripts/run_finetune_GARD_lora.sh \
   /path/to/DIVER-Clinical/scripts/gard/
```

---

## 사용법

### CLI 실행

```bash
cd /path/to/DIVER-Clinical/src

# Task A (Progression Single)
python finetune_main.py \
    --downstream_dataset GARD \
    --gard_task task_a \
    --datasets_dir /pscratch/sd/b/boheelee/GARD/EEG/lmdb/attention/merged_*.lmdb \
    --num_of_classes 2 \
    --batch_size 32 \
    --epochs 50 \
    --use_lora True \
    --lora_r 8 \
    ...

# Task C (Amyloid Prediction)
python finetune_main.py \
    --downstream_dataset GARD \
    --gard_task task_c \
    ...
```

### SLURM 배치

```bash
cd /path/to/DIVER-Clinical/scripts/gard

# 단일 실행
sbatch run_finetune_GARD_lora.sh

# Multi-seed
for seed in 42 123 456; do
    sbatch --export=SEED=$seed run_finetune_GARD_lora.sh
done
```

---

## GARD Task 목록

| Task | 이름 | 샘플 수 | Positive 비율 |
|------|------|--------|--------------|
| task_a | Progression (Single) | ~1,026 | 16% |
| task_b | Progression (Paired) | ~386 | 11% |
| task_c | Amyloid | ~416 | 34% |
| task_d | Hippocampus Atrophy | ~508 | 8% |
| task_e | MTL Atrophy | ~508 | 5% |

---

## LMDB 경로 (Perlmutter)

```
/pscratch/sd/b/boheelee/GARD/EEG/lmdb/
├── attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb
├── beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb
└── sensory/merged_resample-500_highpass-0.3_lowpass-200.lmdb
```

---

## 핵심 차이점

| 항목 | EMBARC | GARD |
|------|--------|------|
| Label 필드 | `data['label']` | `data[self.target_task]` |
| 채널 수 | 19 | 2 (Fp1, Fp2) |
| 태스크 수 | 1 | 5 (task_a ~ task_e) |
| 샘플 필터링 | 전체 사용 | None인 엔트리 제외 |

---

## 문제 해결

### ImportError: gard_dataset

```bash
# gard_dataset.py가 datasets/ 폴더에 있는지 확인
ls /path/to/DIVER-Clinical/src/datasets/gard_dataset.py
```

### KeyError: 'gard'

```python
# generalized_datasets.py의 datasets dict에 'gard' 추가 확인
'gard': (GARDLoadDataset, GARDCustomDataset),
```

### LMDB not found

```bash
# LMDB 경로 확인
ls -la /pscratch/sd/b/boheelee/GARD/EEG/lmdb/attention/
```

---

## 참고 자료

- [GARD DataLoader Guide](../docs/GARD_DataLoader_Guide.md)
- [EMBARC LoRA Script](../../DIVER-Clinical/scripts/embarc/run_finetune_EMBARC_lora.sh)
