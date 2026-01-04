# GARD_EEG

GARD EEG Dataset - neuroNicle FX2 Wireless EEG Data (2019-2023)

---

## 📊 Dataset Summary

- **Period**: 2019 - 2023 (5 years)
- **Total Files**: 10,539 files
- **Subjects**: ~2,663 subjects (based on Excel metadata)
  - ⚠️ Note: Excel list is incomplete, actual number may be higher
- **Device**: Laxtha neuroNicle FX2 (wireless EEG headset)

## 📁 Files by Year

| Year | Number of Files |
|------|-----------------|
| 2019 | 2,134 |
| 2020 | 2,399 |
| 2021 | 1,947 |
| 2022 | 2,095 |
| 2023 | 1,964 |

## 🧠 Channel Configuration (7 channels)

### EEG Channels (2)
- **CH1: Fp1** - Frontal Left EEG
- **CH2: Fp2** - Frontal Right EEG

### Bio Channels (3)
- **CH4: PPG** - Photoplethysmogram
- **CH5: sdPPG** - 2nd derivative PPG
- **CH6: HeartInterval** - RR interval

### Misc Channels (2)
- **CH3: PowerSpectrum** - FFT-derived (updated every 2s)
- **CH7: PacketCounter** - Data packet counter

### Recording Parameters
- **Reference**: Earlobe (monopolar)
- **Sampling Rate**: 250 Hz
- **Hardware Filter**: 3-41 Hz bandpass (built-in)

## 🎯 Task Types (3 protocols per subject)

### 1. BEAM (`*_b_raw.txt`)
- No event file
- Duration: ~5-10 minutes

### 2. Sensory (`*_s_raw.txt`)
- No event file
- Duration: ~5-10 minutes

### 3. Attention (`*_a_raw.txt` + `*_a_raw.event`)
- Event file included
- Duration: ~5-10 minutes
- Events: Stimulus presentations + Response times

## 📌 Event Types (Attention task only)

### Stimulus Events
- **St1, St2, St3, ...** (numbered stimuli)
- **Stimulus types**: "back.wav", "odd.wav" (oddball paradigm)

### Response Events
- **RT**: Reaction time markers

### Example (k_001_oid_5740 attention task)
- Total events: 390
- Duration: 325 seconds (~5.4 minutes)

## 📝 File Naming Convention

**Pattern**: `[gender]_[seq]_oid_[oid]_[task]_[device]_[session]-[date]_[time]([duration])_([channels])_[suffix]_raw.[ext]`

**Example**:
```
k_001_oid_5740_atten_q1n8_id612-2019y-11m_08d_11h_08m_52s(5m_25s)_(l115_r118)_a_raw.txt
```

**Components**:
- `k_001`: Subject sequence
- `oid_5740`: Object ID (subject identifier)
- `atten`: Task type (attention)
- `2019y-11m_08d`: Date
- `(5m_25s)`: Duration
- `(l115_r118)`: Channel values
- `a_raw.txt`: Attention task raw data

## 📋 Metadata

- **Excel file**: `EEG_list_2019-2023.xlsx` (2,663 records)
- **Columns**: 측정년도, k_no, object_idx (OID), Gender, Birth, Age, Device, Device_ID, BEAM파일명, Sensory파일명, Attention파일명

## ⚠️ Notes

- Attention task is the only protocol with event markers
- Excel metadata is incomplete (some OIDs in server files not in Excel)
- Data is already preprocessed (txt format) with hardware filtering applied
- Each subject typically has 3 files (beam, sensory, attention)

---

*Written by Bohee with Claude Code*
