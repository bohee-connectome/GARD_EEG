# LMDB 변환 작업 기록

**작업일**: 2025-01-16
**작업자**: bohee
**상태**: ✅ 완료

---

## 1. 작업 개요

EDF 파일을 DIVER 모델 학습용 LMDB 포맷으로 변환 완료.

### 작업 흐름
```
Lab 서버 (node3)
┌─────────────────────────────────────────────────┐
│ EDF 파일 (2.9GB)                                │
│ /storage/bigdata/GARD/EEG/edf/                  │
│         ↓                                       │
│ LMDB 변환 (8.2GB)                               │
│ /storage/bigdata/GARD/EEG/lmdb/                 │
└─────────────────────────────────────────────────┘
```

---

## 2. 변환 설정

### DIVER 모델 호환 설정
| 항목 | 값 |
|------|-----|
| Resample rate | 500 Hz (원본 250Hz → 업샘플링) |
| Segment length | 30초 |
| Channels | Fp1, Fp2 (EEG only) |
| Filter | 0.3-200Hz bandpass, 60Hz notch |
| Split ratio | 60:20:20 (train:val:test, subject 단위) |

### 출력 형식
- **Shape**: `(2, 30, 500)` = (채널, 초, Hz)
- **dtype**: float32
- **Key format**: `GARD_{year}_{oid}_{task}_seg{NNNN}`

---

## 3. 파일명 패턴 이슈

### 문제 발견
2022년 파일이 처리되지 않는 문제 발생.

### 원인
연도별로 파일명 prefix가 다름:
- 2019-2021, 2023: `k_NNN_oid_XXXXX_task_...edf`
- 2022: `a_NNN_oid_XXXXX_task_...edf`

### 해결
정규식 패턴 수정:
```python
# 변경 전
r'k_(\d+)_oid_(\d+)_(\w+)_.*\.edf'

# 변경 후
r'[a-z]_(\d+)_oid_(\d+)_(\w+)_.*\.edf'
```

---

## 4. 최종 결과

### LMDB 저장 위치
```
/storage/bigdata/GARD/EEG/lmdb/
├── beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb      (2.3GB)
├── sensory/merged_resample-500_highpass-0.3_lowpass-200.lmdb   (3.6GB)
└── attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb (2.3GB)
```

### 샘플 수 (attention 기준)
| Split | Subjects | Samples |
|-------|----------|---------|
| Train | 2,018 | 20,180 |
| Val | 820 | 8,200 |
| Test | 815 | 8,150 |
| **Total** | **3,653** | **36,530** |

### 연도별 분포 확인
| 연도 | Train | Val | Test |
|------|-------|-----|------|
| 2019 | 4,410 | 1,840 | 2,010 |
| 2020 | 5,010 | 2,100 | 2,140 |
| 2021 | 4,080 | 1,790 | 1,630 |
| 2022 | 2,890 | 950 | 960 |
| 2023 | 3,790 | 1,520 | 1,410 |

✅ 전체 연도 (2019-2023) 처리 완료

---

## 5. 실행 명령어

### 스크립트 위치
- 로컬: `~/GARD_EEG/scripts/`
- 서버: `/home/connectome/bohee/GARD_EEG/scripts/`

### 서버 전송
```bash
scp ~/GARD_EEG/scripts/*.py ~/GARD_EEG/scripts/*.sh bohee@147.47.200.154:/home/connectome/bohee/GARD_EEG/scripts/
```

### 변환 실행
```bash
conda activate /storage/connectome/bohee/DIVER_ADFTD/conda_env && cd ~/GARD_EEG/scripts && chmod +x run_preprocessing_gard.sh && ./run_preprocessing_gard.sh all
```

### LMDB 검증
```bash
python check_lmdb_gard.py /storage/bigdata/GARD/EEG/lmdb/beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb
```

---

## 6. 검증 결과

```
======================================================================
🎉 ALL CHECKS PASSED! Dataset is ready for training.
======================================================================

Validation Results:
  ✅ Total samples > 0
  ✅ Valid sample shape
  ✅ No NaN values
  ✅ No Inf values
  ✅ Train samples > 0
  ✅ Val samples > 0
  ✅ Test samples > 0
```

---

## 7. Subject ID 매칭 확인

- 파일명 `oid_XXXXX` = Excel `object_idx`
- LMDB subject ID: `{year}_{oid}` (예: `2019_10027`)

---

## 8. 다음 단계 (TODO)

- [ ] DIVER 모델 finetuning 시작
- [ ] DataLoader 설정 및 학습 파이프라인 구축
- [ ] 성능 평가 (task별 비교)

---

*Written by Bohee with Claude Code*
