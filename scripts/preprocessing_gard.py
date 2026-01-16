#!/usr/bin/env python3
"""
GARD EEG to LMDB Converter
- DIVER 모델 학습용 LMDB 포맷 변환
- Task별 finetune용 (beam, sensory, attention)
- 60:20:20 split (subject 기준)

사용법:
    python preprocessing_gard.py --task beam --data_path /storage/bigdata/GARD/EEG/edf --save_path /storage/bigdata/GARD/EEG/lmdb

작성일: 2025-01-15
"""

import os
import re
import random
import argparse
import pickle
import lmdb
import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
GARD_CHANNELS = ['Fp1', 'Fp2', 'PPG', 'sdPPG', 'HeartInterval', 'PowerSpectrum', 'PacketCounter']
ORIGINAL_SAMPLING_RATE = 250  # GARD EEG 원본 샘플링 레이트
TARGET_SAMPLING_RATE = 500    # DIVER 모델 호환용 (500Hz로 리샘플링)
SEGMENT_LEN = 30     # 30초 세그먼트
TASKS = ['beam', 'sensory', 'attention']

# 전처리 설정 (DIVER 동일)
HIGHPASS = 0.3
LOWPASS = 200
NOTCH = [60]

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='GARD EEG to LMDB Converter')
parser.add_argument('--task', type=str, required=True, choices=TASKS + ['all'],
                    help='Task to process (beam/sensory/attention/all)')
parser.add_argument('--data_path', type=str, default='/storage/bigdata/GARD/EEG/edf',
                    help='Root directory of EDF files')
parser.add_argument('--save_path', type=str, default='/storage/bigdata/GARD/EEG/lmdb',
                    help='Directory to save LMDB files')
parser.add_argument('--resample_rate', type=int, default=500,
                    help='Resample rate (default: 500 for DIVER compatibility)')
parser.add_argument('--segment_len', type=int, default=30,
                    help='Segment length in seconds (default: 30)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for split (default: 42)')
parser.add_argument('--eeg_only', action='store_true',
                    help='Use only EEG channels (Fp1, Fp2)')
parser.add_argument('--debug', action='store_true',
                    help='Debug mode (process only 10 files)')
parser.add_argument('--dry_run', action='store_true',
                    help='Dry run mode (show file list and split info only, no processing)')

args = parser.parse_args()


# ============================================================
# Helper Functions
# ============================================================
def setup_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def get_file_list(root_dir, task):
    """
    Get list of EDF files for a specific task

    Args:
        root_dir: Root directory (e.g., /storage/bigdata/GARD/EEG/edf)
        task: Task name (beam/sensory/attention)

    Returns:
        Dict with subject_id -> file_path mapping
    """
    file_dict = {}
    years = ['2019', '2020', '2021', '2022', '2023']

    for year in years:
        task_dir = os.path.join(root_dir, year, task)
        if not os.path.exists(task_dir):
            print(f"[Warning] Directory not found: {task_dir}")
            continue

        for filename in os.listdir(task_dir):
            if not filename.endswith('.edf'):
                continue

            # Extract subject ID from filename
            # Pattern: k_001_oid_5740_beam_... or a_039_oid_11313_beam_...
            match = re.match(r'[a-z]_(\d+)_oid_(\d+)_(\w+)_.*\.edf', filename)
            if match:
                seq_num, oid, _ = match.groups()
                full_subject_id = f"{year}_{oid}"
                file_path = os.path.join(task_dir, filename)
                file_dict[full_subject_id] = file_path

    print(f"[{task}] Found {len(file_dict)} EDF files")
    return file_dict


def stratified_split(file_dict, seed=42, train_ratio=0.6, val_ratio=0.2):
    """
    Split files into train/val/test by subject

    Args:
        file_dict: Dict with subject_id -> file_path
        seed: Random seed
        train_ratio: Training set ratio (default: 0.6)
        val_ratio: Validation set ratio (default: 0.2)

    Returns:
        Dict with 'train', 'val', 'test' keys containing file paths
    """
    random.seed(seed)

    subjects = list(file_dict.keys())
    random.shuffle(subjects)

    n_total = len(subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    splits = {
        'train': [file_dict[s] for s in train_subjects],
        'val': [file_dict[s] for s in val_subjects],
        'test': [file_dict[s] for s in test_subjects]
    }

    print(f"  Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


def read_edf(file_path):
    """
    Read EDF file using MNE

    Args:
        file_path: Path to EDF file

    Returns:
        MNE Raw object or None if failed
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
        return raw
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return None


def preprocess(raw, highpass=HIGHPASS, lowpass=LOWPASS, notch=NOTCH, eeg_only=False):
    """
    Apply preprocessing filters

    Args:
        raw: MNE Raw object
        highpass: Highpass filter frequency
        lowpass: Lowpass filter frequency
        notch: Notch filter frequencies
        eeg_only: Use only EEG channels (Fp1, Fp2)

    Returns:
        Preprocessed MNE Raw object, original sampling rate
    """
    original_sr = int(raw.info['sfreq'])

    # Channel selection
    available_channels = raw.ch_names
    if eeg_only:
        target_channels = ['Fp1', 'Fp2']
    else:
        target_channels = GARD_CHANNELS

    # Pick available channels
    channels_to_pick = [ch for ch in target_channels if ch in available_channels]
    if not channels_to_pick:
        print(f"[Warning] No target channels found. Available: {available_channels}")
        return None, original_sr

    raw.pick_channels(channels_to_pick)

    # Apply filters (only to EEG channels, skip PPG etc.)
    eeg_channels = [ch for ch in channels_to_pick if ch in ['Fp1', 'Fp2']]
    if eeg_channels:
        # Bandpass filter
        nyquist = original_sr / 2
        if lowpass and lowpass < nyquist:
            raw.filter(l_freq=highpass, h_freq=lowpass, picks=eeg_channels, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=None, picks=eeg_channels, verbose='ERROR')

        # Notch filter
        if notch:
            harmonics = []
            for base in notch:
                mult = 1
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            if harmonics:
                raw.notch_filter(harmonics, picks=eeg_channels, verbose='ERROR')

    return raw, original_sr


def segment_data(raw, segment_len=SEGMENT_LEN, resample_rate=TARGET_SAMPLING_RATE):
    """
    Segment continuous EEG data into fixed-length windows

    Args:
        raw: MNE Raw object
        segment_len: Segment length in seconds
        resample_rate: Target sampling rate

    Returns:
        List of (segment_array, start_time) tuples
    """
    # Get data as numpy array (channels x time)
    data = raw.get_data()
    n_channels, n_samples = data.shape
    original_sr = int(raw.info['sfreq'])

    # Resample if needed
    if original_sr != resample_rate:
        data = mne.filter.resample(data, up=resample_rate, down=original_sr)
        n_samples = data.shape[1]

    # Calculate window size
    window_samples = segment_len * resample_rate
    n_segments = n_samples // window_samples

    if n_segments == 0:
        return []

    segments = []
    for i in range(n_segments):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        segment = data[:, start_idx:end_idx]

        # Reshape to (channels, seconds, samples_per_second)
        # e.g., (2, 30, 500) for 2 channels (Fp1, Fp2), 30 seconds, 500 Hz
        # DIVER 형식: (C, N, P) where N=seconds, P=samples_per_second
        segment_reshaped = segment.reshape(n_channels, segment_len, resample_rate)

        start_time = i * segment_len
        segments.append((segment_reshaped, start_time))

    return segments


def extract_metadata(file_path, segment_idx, task):
    """
    Extract metadata from file path

    Args:
        file_path: Path to EDF file
        segment_idx: Segment index within the file
        task: Task name

    Returns:
        Dict with metadata
    """
    filename = os.path.basename(file_path)
    # Pattern: k_001_oid_5740_beam_... or a_039_oid_11313_beam_...
    match = re.match(r'[a-z]_(\d+)_oid_(\d+)_(\w+)_.*\.edf', filename)

    if match:
        seq_num, oid, _ = match.groups()
        # Extract year from path (e.g., /2019/beam/)
        year = 'unknown'
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if part in ['2019', '2020', '2021', '2022', '2023']:
                year = part
                break
        subject_id = oid
    else:
        year = 'unknown'
        subject_id = filename.replace('.edf', '')

    sample_key = f"GARD_{year}_{subject_id}_{task}_seg{segment_idx:04d}"

    return {
        'sample_key': sample_key,
        'subject_id': f"{year}_{subject_id}",
        'year': year,
        'task': task,
        'segment_index': segment_idx
    }


def process_file(file_path, task, resample_rate, segment_len, eeg_only=False):
    """
    Process a single EDF file

    Args:
        file_path: Path to EDF file
        task: Task name
        resample_rate: Target sampling rate
        segment_len: Segment length in seconds
        eeg_only: Use only EEG channels

    Returns:
        List of (sample_key, value_to_store) tuples
    """
    raw = read_edf(file_path)
    if raw is None:
        return []

    raw, original_sr = preprocess(raw, eeg_only=eeg_only)
    if raw is None:
        return []

    segments = segment_data(raw, segment_len, resample_rate)
    if not segments:
        return []

    results = []
    channel_names = raw.ch_names

    for i, (segment, start_time) in enumerate(segments):
        meta = extract_metadata(file_path, i, task)

        value_to_store = {
            'sample': segment.astype(np.float32),
            'label': None,  # GARD는 분류 레이블 없음 (pretraining용)
            'split': None,  # 나중에 설정
            'data_info': {
                'Dataset': 'GARD',
                'modality': 'EEG',
                'release': meta['year'],
                'subject_id': meta['subject_id'],
                'task': task,
                'resampling_rate': resample_rate,
                'original_sampling_rate': original_sr,
                'segment_index': i,
                'start_time': start_time,
                'channel_names': channel_names,
                'xyz_id': None,  # GARD는 좌표 정보 없음
            }
        }

        results.append((meta['sample_key'], value_to_store))

    return results


def save_to_lmdb(results, split, lmdb_path, map_size=10*1024*1024*1024):
    """
    Save processed data to LMDB

    Args:
        results: List of (sample_key, value_to_store) tuples
        split: Split name ('train', 'val', 'test')
        lmdb_path: Path to LMDB file
        map_size: LMDB map size (default: 10GB)
    """
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=map_size)
    keys = []

    with env.begin(write=True) as txn:
        for sample_key, value in results:
            # Update split info
            value['split'] = split
            txn.put(sample_key.encode(), pickle.dumps(value))
            keys.append(sample_key)

        # Save keys
        txn.put(b'__keys__', pickle.dumps(keys))

    env.close()
    print(f"  Saved {len(keys)} samples to {lmdb_path}")


def merge_lmdb_splits(lmdb_paths, merged_path, map_size=30*1024*1024*1024):
    """
    Merge train/val/test LMDB files into one

    Args:
        lmdb_paths: List of (split_name, lmdb_path) tuples
        merged_path: Path to merged LMDB file
        map_size: LMDB map size (default: 30GB)
    """
    if os.path.exists(merged_path):
        import shutil
        shutil.rmtree(merged_path)

    env_merged = lmdb.open(merged_path, map_size=map_size)
    merged_keys = {'train': [], 'val': [], 'test': []}

    with env_merged.begin(write=True) as txn_merged:
        for split, source_path in lmdb_paths:
            if not os.path.exists(source_path):
                print(f"[Warning] LMDB not found: {source_path}")
                continue

            env_source = lmdb.open(source_path, readonly=True, lock=False)
            with env_source.begin() as txn_source:
                cursor = txn_source.cursor()
                for key, value in cursor:
                    if key == b'__keys__':
                        continue
                    txn_merged.put(key, value)
                    decoded_key = key.decode()
                    merged_keys[split].append(decoded_key)
            env_source.close()

        txn_merged.put(b'__keys__', pickle.dumps(merged_keys))

    env_merged.close()

    total = sum(len(v) for v in merged_keys.values())
    print(f"Merged LMDB: {total} samples (train={len(merged_keys['train'])}, "
          f"val={len(merged_keys['val'])}, test={len(merged_keys['test'])})")


# ============================================================
# Main
# ============================================================
def process_task(task, data_path, save_path, resample_rate, segment_len, seed, eeg_only, debug, dry_run=False):
    """
    Process all files for a single task
    """
    print(f"\n{'='*60}")
    print(f"Processing Task: {task}")
    print(f"{'='*60}")

    # Get file list
    file_dict = get_file_list(data_path, task)
    if not file_dict:
        print(f"[Error] No files found for task: {task}")
        return

    # Split files
    splits = stratified_split(file_dict, seed=seed)

    # Dry run mode - just show info
    if dry_run:
        print(f"\n📋 DRY RUN - File Summary for {task}")
        print(f"{'─'*40}")
        total_files = 0
        for split_name, file_paths in splits.items():
            print(f"  {split_name:5s}: {len(file_paths):4d} files")
            total_files += len(file_paths)
            if len(file_paths) > 0:
                print(f"         Sample: {os.path.basename(file_paths[0])}")
        print(f"{'─'*40}")
        print(f"  Total: {total_files:4d} files")
        print(f"\n📊 Expected Output:")
        print(f"  Shape: ({2 if eeg_only else 7}, {segment_len}, {resample_rate})")
        print(f"  LMDB path: {save_path}/{task}/merged_resample-{resample_rate}_*.lmdb")
        return

    # Process each split
    lmdb_paths = []
    for split_name, file_paths in splits.items():
        if debug:
            file_paths = file_paths[:10]

        print(f"\n[{split_name.upper()}] Processing {len(file_paths)} files...")

        all_results = []
        for file_path in tqdm(file_paths, desc=f"Processing {split_name}"):
            results = process_file(file_path, task, resample_rate, segment_len, eeg_only)
            all_results.extend(results)

        if not all_results:
            print(f"[Warning] No valid segments for {split_name}")
            continue

        # Save to LMDB
        lmdb_name = f"{split_name}_resample-{resample_rate}_highpass-{HIGHPASS}_lowpass-{LOWPASS}.lmdb"
        lmdb_path = os.path.join(save_path, task, lmdb_name)
        save_to_lmdb(all_results, split_name, lmdb_path)
        lmdb_paths.append((split_name, lmdb_path))

    # Merge splits
    if lmdb_paths:
        merged_name = f"merged_resample-{resample_rate}_highpass-{HIGHPASS}_lowpass-{LOWPASS}.lmdb"
        merged_path = os.path.join(save_path, task, merged_name)
        merge_lmdb_splits(lmdb_paths, merged_path)

    print(f"\n✅ Task {task} completed!")


def main():
    print("="*60)
    print("GARD EEG to LMDB Converter")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Save path: {args.save_path}")
    print(f"Task: {args.task}")
    print(f"Resample rate: {args.resample_rate} Hz")
    print(f"Segment length: {args.segment_len} sec")
    print(f"EEG only: {args.eeg_only}")
    print(f"Random seed: {args.seed}")
    print(f"Debug mode: {args.debug}")
    print(f"Dry run: {args.dry_run}")

    setup_seed(args.seed)

    tasks_to_process = TASKS if args.task == 'all' else [args.task]

    for task in tasks_to_process:
        process_task(
            task=task,
            data_path=args.data_path,
            save_path=args.save_path,
            resample_rate=args.resample_rate,
            segment_len=args.segment_len,
            seed=args.seed,
            eeg_only=args.eeg_only,
            debug=args.debug,
            dry_run=args.dry_run
        )

    print("\n" + "="*60)
    print("🎉 All tasks completed!")
    print("="*60)


if __name__ == '__main__':
    main()
