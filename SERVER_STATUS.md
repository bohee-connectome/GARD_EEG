# GARD Server Status

Server architecture and data structure documentation for GARD EEG dataset.

---

## 🖥️ Server Architecture

```
Local PC (Windows)
    ↓ SSH
node3 (147.47.200.154)
    ↓ SSH (port 7777)
nrcd-master (220.67.213.225)
```

### Access Methods

**Step 1**: Connect to node3
```bash
ssh bohee@147.47.200.154
```

**Step 2**: Connect to nrcd-master
```bash
ssh -p 7777 connectome@220.67.213.225
```

---

## 📂 NRCD-MASTER Server (Original Data Location)

**Base Path**: `/lustre/external/connectome/brainwave/`

### Directory Structure

```
/lustre/external/connectome/brainwave/
├── 2019/                          (2,134 files)
│   ├── k_001_oid_5740_beam_*.txt
│   ├── k_001_oid_5740_sens_*.txt
│   ├── k_001_oid_5740_atten_*.txt
│   ├── k_001_oid_5740_atten_*.event
│   └── ... (more subjects)
├── 2020/                          (2,399 files)
├── 2021/                          (1,947 files)
├── 2022/                          (2,095 files)
└── 2023/                          (1,964 files)
```

**Total**: 10,539 files across 5 years
**Total Size**: ~58 GB

---

## 📊 File Distribution Per Subject

Each subject typically has 3 tasks × file types = ~6 files:

### 1. BEAM Task
- `*_b_raw.txt` (EEG data only)

### 2. Sensory Task
- `*_s_raw.txt` (EEG data only)

### 3. Attention Task
- `*_a_raw.txt` (EEG data)
- `*_a_raw.event` (Event markers)

### Estimated Subjects Per Year

| Year | Files | Estimated Subjects |
|------|-------|--------------------|
| 2019 | 2,134 | ~711 |
| 2020 | 2,399 | ~800 |
| 2021 | 1,947 | ~649 |
| 2022 | 2,095 | ~698 |
| 2023 | 1,964 | ~655 |

---

## 🗂️ File Format Analysis

### TXT File Structure

```
Line 1:    Analysis ID:2    Name:Select
Line 2:    Title:
Line 3:    x:Time    y:ch1    y:ch2    y:ch3    y:ch4    y:ch5    y:ch6    y:ch7
Line 4+:   [timestamp]    [ch1_value]    [ch2_value]    ...
```

- **Separator**: TAB
- **Header**: 3 lines
- **Data**: Tab-separated values
- **Sampling**: 0.004s intervals (250 Hz)

### EVENT File Structure (Attention task only)

```
Line 1:    Total_Event    [N]
Line 2:    Event_Pos    Event_Content
Line 3+:   [sample_num]    [event_description]
```

- **Event_Pos**: Sample number (convert to time: sample/250)
- **Event types**: St1-StN (stimuli), RT (response time)
- **Stimulus files**: back.wav, odd.wav (oddball paradigm)

---

## 📏 Data Characteristics

### Duration Per Task
Typically ~5-10 minutes (varies by subject/task)

**Example durations** (k_001):
- BEAM: 310 seconds (5m 10s)
- Sensory: 485 seconds (8m 5s)
- Attention: 325 seconds (5m 25s)

### File Sizes (estimated from samples)
- txt files: ~2-4 MB per file (depends on duration)
- event files: ~10-20 KB per file

### Total Dataset Size
- **Original txt files**: ~58 GB (measured)
- **Estimated FIF files**: ~12-15 GB (after conversion with compression)

---

## 🔍 Metadata File

**Location**: Downloaded to local PC
**File**: `C:\Users\user\Downloads\EEG_list_2019-2023.xlsx`

**Records**: 2,663 rows

### Columns
- 측정년도 (Measurement Year)
- k_no (Subject sequence number)
- object_idx (OID - Subject ID)
- Gender
- Birth (Birth year)
- Age
- Device
- Device_ID
- BEAM파일명 (BEAM filename)
- Sensory파일명 (Sensory filename)
- Attention파일명 (Attention filename)

### ⚠️ Known Issues
- Excel is incomplete
- Some OIDs in server don't exist in Excel
- Example: oid_2458 found in 2019 directory but not in Excel

---

## 📁 NODE3 Server (Destination)

**Planned Path**: `/storage/bigdata/GARD/EEG/`

### Proposed Structure

```
/storage/bigdata/GARD/EEG/
├── raw/                           (원본 txt 파일)
│   ├── 2019/
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   └── 2023/
└── processed/                     (변환된 FIF 파일)
    ├── 2019/
    ├── 2020/
    ├── 2021/
    ├── 2022/
    └── 2023/
```

**Status**: 📦 Not yet transferred

---

## 🔄 Current Workflow

### ✅ Completed

1. **Sample data downloaded** to local PC
   - Location: `C:\Users\user\Downloads\sample_eeg\`
   - Subject: k_001_oid_5740
   - Files: 3 txt + 1 event file

2. **Conversion scripts created**
   - `convert_eeg_to_mne.py` (single file conversion)
   - `batch_convert_eeg.py` (batch processing)
   - `run_conversion.sh` (automated pipeline)

3. **Test conversion completed**
   - Input: `sample_eeg/*.txt`
   - Output: `sample_eeg/converted/*.fif`
   - Result: Successfully converted to MNE-Python format

### ⏳ Pending

4. **Full dataset transfer & conversion**
   - Plan: nrcd-master → node3 → process → store

---

## 💾 Storage Requirements

| Item | Size |
|------|------|
| Original txt files | ~58 GB |
| Converted FIF files | ~12-15 GB (estimated) |
| Working space needed | ~70-80 GB (for temporary processing) |

**Recommendation**: Process year-by-year to manage disk space

---

## 🛠️ Disk Usage Commands

### Check available space on nrcd-master
```bash
df -h /lustre/external/connectome/brainwave
```

### Check available space on node3
```bash
df -h /storage/bigdata/GARD
```

### Check current directory size
```bash
du -sh
```

### Check subdirectory sizes
```bash
du -sh *
```

---

*Last updated: 2026-01-05*
*Written by Bohee with Claude Code*
