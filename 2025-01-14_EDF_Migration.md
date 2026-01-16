# EDF 변환 및 Lab 서버 이전 작업 기록

**작업일**: 2025-01-14
**작업자**: bohee
**상태**: ✅ 완료

---

## 1. 작업 개요

GARD 서버(nrcd-master)에 있는 원본 TXT 파일을 EDF 포맷으로 변환하고, Lab 서버(node3)로 이전하는 작업을 완료했습니다.

### 작업 흐름
```
GARD 서버 (nrcd-master)          Lab 서버 (node3)
┌─────────────────────┐          ┌─────────────────────┐
│ 원본 TXT (58GB)     │          │                     │
│         ↓           │          │                     │
│ EDF 변환 (2.9GB)    │ ──rsync──▶ EDF 저장 (2.9GB)   │
│ /home/connectome/   │          │ /storage/bigdata/   │
│ gard_edf/           │          │ GARD/EEG/edf/       │
└─────────────────────┘          └─────────────────────┘
```

---

## 2. GARD 서버에서 EDF 변환

### 2.1 변환 스크립트
**경로**: `/home/connectome/convert_txt_to_edf.py`

```python
from pathlib import Path
import argparse
import mne
import numpy as np
from tqdm import tqdm

def convert_txt_to_edf(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_path.glob("*.txt"))
    print(f"Found {len(txt_files)} TXT files")

    ch_names = ['Fp1', 'Fp2', 'PPG', 'sdPPG', 'HeartInterval', 'PowerSpectrum', 'PacketCounter']
    ch_types = ['eeg', 'eeg', 'misc', 'misc', 'misc', 'misc', 'misc']
    sfreq = 250

    for txt_file in tqdm(txt_files, desc="Converting"):
        try:
            data = np.loadtxt(txt_file, skiprows=3)
            if data.shape[1] >= 7:
                eeg_data = data[:, :7].T
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                raw = mne.io.RawArray(eeg_data, info)
                out_name = txt_file.stem + '.edf'
                out_path = output_path / out_name
                mne.export.export_raw(str(out_path), raw, fmt='edf', overwrite=True)
        except Exception as e:
            print(f"Error: {txt_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_txt_to_edf(args.input, args.output)
```

### 2.2 연도별 변환 명령어

```bash
# 2023년
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2023/raw_txt_2023_final_20240725/beam" && OUTPUT="/home/connectome/gard_edf/2023/beam" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2023/raw_txt_2023_final_20240725/sensory" && OUTPUT="/home/connectome/gard_edf/2023/sensory" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2023/raw_txt_2023_final_20240725/attention" && OUTPUT="/home/connectome/gard_edf/2023/attention" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"

# 2022년
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2022/CRF_matching_2022_final/beam" && OUTPUT="/home/connectome/gard_edf/2022/beam" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2022/CRF_matching_2022_final/sensory" && OUTPUT="/home/connectome/gard_edf/2022/sensory" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2022/CRF_matching_2022_final/attention" && OUTPUT="/home/connectome/gard_edf/2022/attention" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"

# 2021년
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2021/beam" && OUTPUT="/home/connectome/gard_edf/2021/beam" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2021/sensory" && OUTPUT="/home/connectome/gard_edf/2021/sensory" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2021/attention" && OUTPUT="/home/connectome/gard_edf/2021/attention" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"

# 2020년
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2020/beam" && OUTPUT="/home/connectome/gard_edf/2020/beam" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2020/sensory" && OUTPUT="/home/connectome/gard_edf/2020/sensory" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2020/attention" && OUTPUT="/home/connectome/gard_edf/2020/attention" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"

# 2019년 (폴더명 다름)
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2019/2019-beam" && OUTPUT="/home/connectome/gard_edf/2019/beam" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2019/2019-sensory" && OUTPUT="/home/connectome/gard_edf/2019/sensory" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
INPUT="/lustre/external/connectome/brainwave/EEG rawdata_2019/2019-attention" && OUTPUT="/home/connectome/gard_edf/2019/attention" && python /home/connectome/convert_txt_to_edf.py --input "$INPUT" --output "$OUTPUT"
```

### 2.3 Event 파일 복사 (변환 없이 그대로)

```bash
# Event 파일 디렉토리 생성 및 복사
mkdir -p /home/connectome/gard_edf/event/2022 /home/connectome/gard_edf/event/2021 /home/connectome/gard_edf/event/2020 && cp "/lustre/external/connectome/brainwave/EEG rawdata_2022/CRF_matching_2022_final/event/"* /home/connectome/gard_edf/event/2022/ && cp "/lustre/external/connectome/brainwave/EEG rawdata_2021/attention_event/"* /home/connectome/gard_edf/event/2021/ && cp "/lustre/external/connectome/brainwave/EEG rawdata_2020/attention_event/"* /home/connectome/gard_edf/event/2020/
```

---

## 3. Lab 서버로 이전

### 3.1 rsync 명령어 (node3에서 실행)

```bash
# EDF 파일 전송
rsync -avz --progress -e "ssh -p 7777" connectome@220.67.213.225:/home/connectome/gard_edf/ /storage/bigdata/GARD/EEG/edf/

# Event 파일 전송
rsync -avz --progress -e "ssh -p 7777" connectome@220.67.213.225:/home/connectome/gard_edf/event/ /storage/bigdata/GARD/EEG/edf/event/

# Excel 메타데이터 파일 전송
rsync -avz -e "ssh -p 7777" connectome@220.67.213.225:/home/connectome/EEG_list_2019-2023.xlsx /storage/bigdata/GARD/EEG/
```

### 3.2 최종 저장 경로 (Lab 서버)

```
/storage/bigdata/GARD/EEG/
├── edf/
│   ├── 2019/
│   │   ├── beam/       (533개)
│   │   ├── sensory/    (532개)
│   │   └── attention/  (531개)
│   ├── 2020/
│   │   ├── beam/       (601개)
│   │   ├── sensory/    (601개)
│   │   └── attention/  (598개)
│   ├── 2021/
│   │   ├── beam/       (486개)
│   │   ├── sensory/    (486개)
│   │   └── attention/  (487개)
│   ├── 2022/
│   │   ├── beam/       (523개)
│   │   ├── sensory/    (521개)
│   │   └── attention/  (523개)
│   ├── 2023/
│   │   ├── beam/       (490개)
│   │   ├── sensory/    (491개)
│   │   └── attention/  (490개)
│   └── event/
│       ├── 2020/       (598개)
│       ├── 2021/       (487개)
│       └── 2022/       (523개)
└── EEG_list_2019-2023.xlsx
```

---

## 4. 파일 수 검증

### 4.1 변환 결과 (EDF)

| 연도 | beam | sensory | attention | 합계 |
|------|------|---------|-----------|------|
| 2019 | 533 | 532 | 531 | 1,596 |
| 2020 | 601 | 601 | 598 | 1,800 |
| 2021 | 486 | 486 | 487 | 1,459 |
| 2022 | 523 | 521 | 523 | 1,567 |
| 2023 | 490 | 491 | 490 | 1,471 |
| **총** | **2,633** | **2,631** | **2,629** | **7,893** |

### 4.2 Event 파일 (변환 없음)

| 연도 | 원본 | 복사 |
|------|------|------|
| 2020 | 599 | 598 |
| 2021 | 487 | 487 |
| 2022 | 524 | 523 |
| **총** | **1,610** | **1,608** |

### 4.3 원본 TXT vs 변환 EDF 비교

| 연도 | 원본 TXT | 변환 EDF | 차이 |
|------|----------|----------|------|
| 2019 | 1,601 | 1,596 | -5 |
| 2020 | 1,800 | 1,800 | 0 |
| 2021 | 1,460 | 1,459 | -1 |
| 2022 | 1,571 | 1,567 | -4 |
| 2023 | 1,474 | 1,471 | -3 |
| **총** | **7,906** | **7,893** | **-13** |

### 4.4 Excel 메타데이터 vs 실제 파일

**Excel 파일 (EEG_list_2019-2023.xlsx)**:
- 총 세션: 2,663개
- 예상 파일 수: 2,663 × 3 = 7,989개

**검증 결과**:
```bash
# Lab 서버에서 실행
python3 -c "import pandas as pd; df = pd.read_excel('/storage/bigdata/GARD/EEG/EEG_list_2019-2023.xlsx'); print(df['측정년도(데이터년도)'].value_counts().sort_index())"
```
```
2019    539
2020    615
2021    509
2022    500
2023    500
```

**누락 확인 결과**: 3개 (xlsx에 실제 파일명 대신 메모 기록)
- `2022/BEAM파일명: 2022-11-21`
- `2022/BEAM파일명: 측정된 데이터 없음`
- `2022/BEAM파일명: 2022-12-14`

---

## 5. 최종 결과

### 5.1 전송 완료

| 항목 | 파일 수 | 용량 |
|------|---------|------|
| EDF 파일 | 7,893개 | ~2.9GB |
| Event 파일 | 1,608개 | ~24MB |
| **총** | **9,501개** | **~2.9GB** |

### 5.2 압축률
- 원본 TXT: ~58GB
- 변환 EDF: ~2.9GB
- **압축률: 약 20배**

### 5.3 누락 파일 분석

| 구분 | 개수 | 원인 |
|------|------|------|
| 원본 TXT → EDF 변환 실패 | 13개 | 빈 파일 또는 손상된 파일 |
| Event 파일 누락 | 2개 | 미확인 |
| Excel 등록 vs 실제 파일 | 3개 | xlsx에 메모 기록 (실제 파일 없음) |

---

## 6. 검증 명령어 모음

### GARD 서버 (nrcd-master)
```bash
# 변환된 EDF 파일 수 확인
echo "=== 2023 ===" && ls /home/connectome/gard_edf/2023/beam | wc -l && ls /home/connectome/gard_edf/2023/sensory | wc -l && ls /home/connectome/gard_edf/2023/attention | wc -l && echo "=== 2022 ===" && ls /home/connectome/gard_edf/2022/beam | wc -l && ls /home/connectome/gard_edf/2022/sensory | wc -l && ls /home/connectome/gard_edf/2022/attention | wc -l && echo "=== 2021 ===" && ls /home/connectome/gard_edf/2021/beam | wc -l && ls /home/connectome/gard_edf/2021/sensory | wc -l && ls /home/connectome/gard_edf/2021/attention | wc -l && echo "=== 2020 ===" && ls /home/connectome/gard_edf/2020/beam | wc -l && ls /home/connectome/gard_edf/2020/sensory | wc -l && ls /home/connectome/gard_edf/2020/attention | wc -l && echo "=== 2019 ===" && ls /home/connectome/gard_edf/2019/beam | wc -l && ls /home/connectome/gard_edf/2019/sensory | wc -l && ls /home/connectome/gard_edf/2019/attention | wc -l && echo "=== 총 용량 ===" && du -sh /home/connectome/gard_edf/

# 원본 TXT 파일 수 확인
echo "=== 원본 TXT ===" && find "/lustre/external/connectome/brainwave/EEG rawdata_2023" -name "*.txt" | wc -l && find "/lustre/external/connectome/brainwave/EEG rawdata_2022" -name "*.txt" | wc -l && find "/lustre/external/connectome/brainwave/EEG rawdata_2021" -name "*.txt" | wc -l && find "/lustre/external/connectome/brainwave/EEG rawdata_2020" -name "*.txt" | wc -l && find "/lustre/external/connectome/brainwave/EEG rawdata_2019" -name "*.txt" | wc -l
```

### Lab 서버 (node3)
```bash
# 전송된 파일 확인
du -sh /storage/bigdata/GARD/EEG/edf/ && find /storage/bigdata/GARD/EEG/edf/ -type f | wc -l

# Excel 기준 누락 파일 확인
python3 -c "import pandas as pd; import os; import glob; df = pd.read_excel('/storage/bigdata/GARD/EEG/EEG_list_2019-2023.xlsx'); edf_base = '/storage/bigdata/GARD/EEG/edf'; all_edfs = set([os.path.basename(f).lower() for f in glob.glob(f'{edf_base}/**/*.edf', recursive=True)]); missing = []; [missing.append(f\"{row['측정년도(데이터년도)']}/{task}: {str(row[task])[:50]}\") for _, row in df.iterrows() for task in ['BEAM파일명','Sensory파일명','Attention파일명'] if pd.notna(row[task]) and not any(str(row[task]).lower().split('(')[0] in e for e in all_edfs)]; print(f'Missing: {len(missing)}'); [print(m) for m in missing[:30]]"
```

---

## 7. 다음 단계 (TODO)

- [x] LMDB 변환 (딥러닝 학습용) → **완료 (2025-01-16)**
- [ ] GARD 서버 임시 파일 정리 (`/home/connectome/gard_edf/` 삭제)
- [ ] 변환 실패 13개 파일 원인 분석
- [ ] Event 파일 누락 2개 확인

---

## 8. 참고: 주요 경로 요약

| 서버 | 용도 | 경로 |
|------|------|------|
| GARD (nrcd-master) | 원본 TXT | `/lustre/external/connectome/brainwave/EEG rawdata_YYYY/` |
| GARD (nrcd-master) | 변환 EDF (임시) | `/home/connectome/gard_edf/` |
| GARD (nrcd-master) | 변환 스크립트 | `/home/connectome/convert_txt_to_edf.py` |
| GARD (nrcd-master) | Excel 원본 | `/lustre/external/connectome/brainwave/EEG_list_2019-2023.xlsx` |
| Lab (node3) | EDF 저장 | `/storage/bigdata/GARD/EEG/edf/` |
| Lab (node3) | Excel 복사본 | `/storage/bigdata/GARD/EEG/EEG_list_2019-2023.xlsx` |
