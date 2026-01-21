# GARD Custom DataLoader 가이드

## 개요

DIVER 파인튜닝을 위한 GARD 데이터셋 연동 가이드. GARD는 멀티태스크 구조라 Custom DataLoader가 필요함.

---

## LMDB 구조 비교

| 필드 | DIVER 기대값 | EMBARC | LEAD-ADFTD | GARD |
|------|-------------|--------|------------|------|
| `sample` | ✅ ndarray | ✅ (19, 30, 500) | ✅ (19, N, 500) | ✅ (2, 30, 500) |
| `label` | ✅ **int (필수)** | ✅ 0/1 | ✅ 0/1 | ❌ 없음 |
| `split` | ✅ str | ✅ | ✅ | ✅ |
| `task_a~e` | - | - | - | ✅ 0/1/None |
| `data_info` | ✅ dict | ✅ | ✅ | ✅ |

---

## GARD LMDB 구조

```python
{
    'sample': ndarray (2, 30, 500),   # Fp1, Fp2 채널
    'task_a': 0/1/None,               # Progression (Single)
    'task_b': 0/1/None,               # Progression (Paired)
    'task_c': 0/1/None,               # Amyloid Prediction
    'task_d': 0/1/None,               # Hippocampus Atrophy
    'task_e': 0/1/None,               # MTL Cortical Atrophy
    'split': 'train/val/test',
    'data_info': {
        'Dataset': 'GARD',
        'subject_id': '2019_10000',
        'oid': 10000,
        'eeg_task': 'attention',
        ...
    }
}
```

---

## Task 정의

| Task | 이름 | 설명 | 양성 비율 |
|------|------|------|----------|
| task_a | Progression (Single) | CN→MCI 진행 예측 | 16% |
| task_b | Progression (Paired) | 두 시점 EEG 변화 | 11% |
| task_c | Amyloid | Aβ+ vs Aβ- 예측 | 34% |
| task_d | Hippocampus | 해마 위축 예측 | 8% |
| task_e | MTL Atrophy | MTL 피질 위축 예측 | 5% |

---

## Custom DataLoader 구현

### 추가 위치
```
/path/to/DIVER/CBraMod/datasets/gard_dataset.py  (신규 생성)
```

### 코드

```python
import lmdb
import pickle
import torch
from torch.utils.data import Dataset


class GARDCustomDataset(Dataset):
    """GARD 멀티태스크용 Custom Dataset"""

    def __init__(self, datasets_dir, mode='train', target_task='task_c',
                 transform=None, return_info=False):
        """
        Args:
            datasets_dir: LMDB 경로
            mode: 'train', 'val', 'test'
            target_task: 'task_a' | 'task_b' | 'task_c' | 'task_d' | 'task_e'
            transform: 데이터 변환
            return_info: data_info 반환 여부
        """
        self.db = lmdb.open(datasets_dir, readonly=True, lock=False)
        self.mode = mode
        self.target_task = target_task
        self.transform = transform
        self.return_info = return_info

        # 해당 task의 label이 None이 아닌 key만 필터링
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))[self.mode]

            self.keys = []
            for key in all_keys:
                data = pickle.loads(txn.get(key.encode()))
                if data.get(self.target_task) is not None:
                    self.keys.append(key)

        print(f"[GARD] {mode} set: {len(self.keys)}/{len(all_keys)} samples with {target_task}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            data = pickle.loads(txn.get(key.encode()))

        sample = data['sample']
        label = data[self.target_task]  # 핵심: task_X에서 label 읽음
        data_info = data.get('data_info', {})

        sample = torch.from_numpy(sample).float()
        if self.transform:
            sample = self.transform(sample)

        if self.return_info:
            return sample, label, data_info
        return sample, label
```

### 사용법

```python
# Amyloid (task_c) 학습
train_set = GARDCustomDataset(
    '/pscratch/sd/b/boheelee/GARD/EEG/lmdb/attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb',
    mode='train',
    target_task='task_c'
)

# Hippocampus (task_d) 학습 - 파라미터만 변경
train_set = GARDCustomDataset(
    lmdb_path,
    mode='train',
    target_task='task_d'
)
```

---

## 핵심 변경점

| 항목 | 기존 DataLoader | GARD Custom |
|------|----------------|-------------|
| label 읽기 | `data['label']` | `data[self.target_task]` |
| entry 필터링 | 전체 사용 | None인 entry 제외 |
| task 전환 | 불가 | 파라미터로 선택 |

---

## LMDB 경로 (펄머터)

```
/pscratch/sd/b/boheelee/GARD/EEG/lmdb/
├── attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb
├── beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb
└── sensory/merged_resample-500_highpass-0.3_lowpass-200.lmdb
```

### 파일명 규칙
- `resample-500`: 샘플링 레이트 500Hz
- `highpass-0.3`: 하이패스 필터 0.3Hz
- `lowpass-200`: 로우패스 필터 200Hz

---

## 체크리스트

- [x] LMDB 라벨 업데이트 완료 (task_a ~ task_e)
- [ ] GARDCustomDataset 클래스 추가
- [ ] 파인튜닝 테스트 실행
- [ ] 결과 DIVER-Clinical 리더보드에 추가
