# GARD EEG Task Labels

DIVER-Clinical 프로젝트를 위한 GARD EEG 데이터셋 라벨링 정의

---

## Quick Summary

| Task | 이름 | N | 목표 | 세부 내용 | 비율 |
|------|------|---|------|----------|------|
| **A** | Progression (Single) | 1,026 | Baseline EEG → 미래 악화 예측 | CN→SCD/MCI, SCD→MCI, MCI→Dem | 16% prog |
| **B** | Progression (Paired) | 188 | 두 시점 EEG → 진단 변화 | T1: CN/SCD/MCI → T2: 악화 여부 | 11% prog |
| **C** | Amyloid Prediction | 1,524 | EEG → Amyloid PET 예측 | Aβ+ vs Aβ- (SUVR cutoff 기반) | 34% pos |
| **D** | Hippocampus Atrophy | 2,130 | EEG → 해마 위축 예측 | HV/ICV < 4.60 (CN 5%ile, 문헌표준) | 8% atrophy |
| **E** | MTL Cortical Atrophy | 1,940 | EEG → MTL 위축 예측 | Entorhinal+Parahippocampal < 2.58mm | 5% atrophy |

### 태스크 특성 비교

| 특성 | A/B | C | D/E |
|------|-----|---|-----|
| **예측 대상** | 임상 진단 변화 | PET 바이오마커 | MRI 구조적 위축 |
| **시간 관계** | 미래 예측 | 동시점 | 동시점 |
| **임상 의미** | 질병 진행 | 병리 존재 | 신경퇴행 |
| **초기 감별** | 중간 | 높음 | D<E (E가 더 초기) |

### 파일 목록
```
labels/
├── task_a_progression_single.csv
├── task_b_progression_paired.csv
├── task_c_amyloid.csv
├── task_d_hippocampus.csv
└── task_e_mtl_atrophy.csv
```

---

## Task Details

### Task A: Progression Prediction (Single EEG)

**목표**: Baseline EEG만으로 향후 인지기능 악화 예측

| 항목 | 내용 |
|------|------|
| Input | Single EEG (baseline) |
| Output | Binary (progressive / stable) |
| N | 1,026 |
| Progressive | 165 (16%) |
| Stable | 861 (84%) |

**Progressive 정의**:
- CN → SCD, MCI, Dem
- SCD → MCI, Dem
- MCI → Dem

**파일**: `task_a_progression_single.csv`

---

### Task B: Progression Prediction (Paired EEG)

**목표**: 두 시점 EEG로 진단 변화 예측

| 항목 | 내용 |
|------|------|
| Input | Paired EEG (T1, T2) |
| Output | Binary (progressive / stable) |
| N | 188 |
| Progressive | 21 (11%) |
| Stable | 167 (89%) |

**조건**: 동일 피험자의 서로 다른 연도 EEG 측정

**파일**: `task_b_progression_paired.csv`

---

### Task C: Amyloid Prediction

**목표**: EEG로 뇌 아밀로이드 축적 여부 예측

| 항목 | 내용 |
|------|------|
| Input | EEG |
| Output | Binary (Aβ+ / Aβ-) |
| N | 1,524 |
| Positive | 511 (34%) |
| Negative | 1,013 (66%) |

**Cutoff**: SUVR 기반 (임상 PET 판독 결과)

**파일**: `task_c_amyloid.csv`

---

### Task D: Hippocampus Atrophy Prediction

**목표**: EEG로 해마 위축 여부 예측

| 항목 | 내용 |
|------|------|
| Input | EEG |
| Output | Binary (atrophy / normal) |
| N | 2,130 |
| Atrophy | 162 (8%) |
| Normal | 1,968 (92%) |

**Cutoff**: HV/ICV×1000 < **4.60**
- 근거: CN 5th percentile (Jack et al. 문헌 표준)
- GARD CN 분포와 문헌 값 일치 확인

**파일**: `task_d_hippocampus.csv`

---

### Task E: MTL Cortical Thickness Atrophy

**목표**: EEG로 내측두엽 피질 위축 예측

| 항목 | 내용 |
|------|------|
| Input | EEG |
| Output | Binary (atrophy / normal) |
| N | 1,940 |
| Atrophy | 93 (5%) |
| Normal | 1,847 (95%) |

**측정 영역**: (L_entorhinal + R_entorhinal + L_parahippocampal + R_parahippocampal) / 4

**Cutoff**: MTL thickness < **2.58 mm**
- 근거: 5th percentile 기준
- Entorhinal cortex는 AD 초기 변화가 가장 먼저 나타나는 영역 (Leal & Yassa, 2013)

**파일**: `task_e_mtl_atrophy.csv`

---

## Diagnosis Hierarchy

```
CN (0) → SCD (1) → MCI (2) → Dem (3)
```

| 약어 | 의미 |
|------|------|
| CN | Cognitively Normal |
| SCD | Subjective Cognitive Decline |
| MCI | Mild Cognitive Impairment |
| Dem | Dementia |

---

## CSV Column Description

### Common Columns

| Column | Description |
|--------|-------------|
| `object_idx` | GARD 원본 인덱스 |
| `subject_id` | 피험자 ID |
| `eeg_years` | EEG 측정 연도 |
| `label` | 0 또는 1 |
| `label_name` | 라벨 텍스트 |

### Task-specific Columns

| Task | Additional Columns |
|------|--------------------|
| A | `diagnosis_baseline`, `diagnosis_final`, `years_to_final` |
| B | `eeg_t1_year`, `eeg_t2_year`, `delta_t`, `diagnosis_t1`, `diagnosis_t2` |
| C | `amyloid_status`, `suvr` (if available) |
| D | `hippo_normalized`, `hippo_left`, `hippo_right`, `icv` |
| E | `mtl_thickness` |

---

## LMDB-CSV ID 매핑

### ID 체계

| 소스 | 필드명 | 형식 | 예시 |
|------|--------|------|------|
| **LMDB** | `data_info['subject_id']` | `{year}_{oid}` | `2019_23` |
| **CSV** | `object_idx` | 정수 | `23` |
| **CSV** | `subject_id` | `GM{idx:05d}` | `GM00023` |

### 매핑 방법

```python
# LMDB → CSV 매핑
lmdb_subject_id = data['data_info']['subject_id']  # "2019_23"
oid = int(lmdb_subject_id.split('_')[1])  # 23
# CSV에서 object_idx == oid 인 행의 label 사용

# 예시
# LMDB: subject_id="2019_23" → oid=23
# CSV: object_idx=23, subject_id="GM00023", label=0
```

### LMDB 구조

```
/pscratch/sd/b/boheelee/GARD/EEG/lmdb/
├── attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb
├── beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb
└── sensory/merged_resample-500_highpass-0.3_lowpass-200.lmdb
```

**LMDB 데이터 구조**:
- `sample`: ndarray (2, 30, 500) - Fp1, Fp2 채널
- `label`: None → CSV에서 매핑 필요
- `split`: str (train/val/test)
- `data_info`: dict (subject_id, task, year 등)

### 주의사항

1. LMDB의 `task` (attention/beam/sensory)는 **EEG 측정 패러다임**
2. CSV의 Task A-E는 **분류 태스크** (별개 개념)
3. 한 subject가 여러 segment 가질 수 있음 (seg0000, seg0001, ...)

---

## References

- Jack et al. - Hippocampal volume normative cutoffs
- Leal & Yassa (2013) - Entorhinal cortex as earliest AD biomarker
- Pettigrew et al. (2016) - Cortical thickness and AD progression
