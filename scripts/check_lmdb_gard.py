#!/usr/bin/env python3
"""
GARD LMDB Verification Script
- LMDB 파일 구조 및 데이터 검증

사용법:
    python check_lmdb_gard.py /path/to/merged.lmdb
    python check_lmdb_gard.py /path/to/merged.lmdb --detailed

작성일: 2025-01-15
"""

import sys
import lmdb
import pickle
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description='GARD LMDB Verification')
parser.add_argument('lmdb_path', type=str, help='Path to LMDB file')
parser.add_argument('--detailed', action='store_true', help='Show detailed sample info')

args = parser.parse_args()

# ============================================================
# Main Verification
# ============================================================
print("="*70)
print("GARD LMDB Verification")
print("="*70)
print(f"LMDB path: {args.lmdb_path}")

try:
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
except Exception as e:
    print(f"\n❌ Failed to open LMDB: {e}")
    sys.exit(1)

with env.begin() as txn:
    # ============================================================
    # 1. Get Dataset Keys
    # ============================================================
    keys_data = txn.get(b'__keys__')
    if not keys_data:
        print("\n❌ No dataset keys found in LMDB!")
        env.close()
        sys.exit(1)

    dataset = pickle.loads(keys_data)

    # Check if it's a merged LMDB (dict) or single split (list)
    if isinstance(dataset, dict):
        is_merged = True
        all_keys = []
        for split_keys in dataset.values():
            all_keys.extend(split_keys)
    else:
        is_merged = False
        all_keys = dataset
        dataset = {'all': dataset}

    # ============================================================
    # 2. Dataset Split Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("[1] Dataset Summary")
    print(f"{'='*70}")
    print(f"  Type: {'Merged (train/val/test)' if is_merged else 'Single split'}")

    if is_merged:
        for split in ['train', 'val', 'test']:
            if split in dataset:
                print(f"  {split.capitalize():8s}: {len(dataset[split]):,} samples")

    print(f"  {'─'*40}")
    print(f"  Total:    {len(all_keys):,} samples")

    # ============================================================
    # 3. Sample Structure Verification
    # ============================================================
    print(f"\n{'='*70}")
    print("[2] Sample Structure")
    print(f"{'='*70}")

    if is_merged:
        splits_to_check = ['train', 'val', 'test']
    else:
        splits_to_check = ['all']

    for split in splits_to_check:
        if split not in dataset or not dataset[split]:
            continue

        first_key = dataset[split][0]
        sample_data = pickle.loads(txn.get(first_key.encode()))

        print(f"\n{split.upper()} Sample:")
        print(f"  Key: {first_key}")
        print(f"  Sample shape: {sample_data['sample'].shape}")
        print(f"  Sample dtype: {sample_data['sample'].dtype}")
        print(f"  Label: {sample_data['label']}")
        print(f"  Subject: {sample_data['data_info']['subject_id']}")
        print(f"  Task: {sample_data['data_info']['task']}")
        print(f"  Channels: {sample_data['data_info']['channel_names']}")
        print(f"  Resampling rate: {sample_data['data_info']['resampling_rate']} Hz")

        # Shape validation for GARD
        shape = sample_data['sample'].shape
        n_channels = shape[0]
        expected_shape_pattern = (n_channels, 30, sample_data['data_info']['resampling_rate'])

        if shape == expected_shape_pattern:
            print(f"  ✅ Shape correct: {shape}")
        else:
            print(f"  ⚠️  Shape: {shape} (expected pattern: (channels, 30, sr))")

    # ============================================================
    # 4. Subject Distribution
    # ============================================================
    print(f"\n{'='*70}")
    print("[3] Subject Distribution")
    print(f"{'='*70}")

    for split in splits_to_check:
        if split not in dataset or not dataset[split]:
            continue

        subjects = set()
        tasks = Counter()
        years = Counter()

        for key in dataset[split]:
            data = pickle.loads(txn.get(key.encode()))
            subjects.add(data['data_info']['subject_id'])
            tasks[data['data_info']['task']] += 1
            years[data['data_info']['release']] += 1

        print(f"\n{split.upper()} split:")
        print(f"  Unique subjects: {len(subjects)}")
        print(f"  Tasks: {dict(tasks)}")
        print(f"  Years: {dict(sorted(years.items()))}")

    # ============================================================
    # 5. Data Value Statistics
    # ============================================================
    print(f"\n{'='*70}")
    print("[4] Data Value Statistics")
    print(f"{'='*70}")

    # Sample a few records for statistics
    n_samples_to_check = min(100, len(all_keys))
    sample_keys = all_keys[:n_samples_to_check]

    all_values = []
    for key in sample_keys:
        data = pickle.loads(txn.get(key.encode()))
        all_values.append(data['sample'])

    all_values = np.array(all_values)
    print(f"\n  Checked {n_samples_to_check} samples:")
    print(f"  Overall shape: {all_values.shape}")
    print(f"  Mean: {np.mean(all_values):.6f}")
    print(f"  Std:  {np.std(all_values):.6f}")
    print(f"  Min:  {np.min(all_values):.6f}")
    print(f"  Max:  {np.max(all_values):.6f}")

    # Check for NaN/Inf
    n_nan = np.sum(np.isnan(all_values))
    n_inf = np.sum(np.isinf(all_values))
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠️  NaN values: {n_nan}")
        print(f"  ⚠️  Inf values: {n_inf}")
    else:
        print(f"  ✅ No NaN/Inf values")

    # ============================================================
    # 6. Detailed Sample Info (optional)
    # ============================================================
    if args.detailed:
        print(f"\n{'='*70}")
        print("[5] Detailed Sample Info (first 5 samples)")
        print(f"{'='*70}")

        for i, key in enumerate(all_keys[:5]):
            data = pickle.loads(txn.get(key.encode()))
            print(f"\n  Sample {i+1}:")
            print(f"    Key: {key}")
            print(f"    Subject: {data['data_info']['subject_id']}")
            print(f"    Task: {data['data_info']['task']}")
            print(f"    Year: {data['data_info']['release']}")
            print(f"    Segment: {data['data_info']['segment_index']}")
            print(f"    Start time: {data['data_info']['start_time']} sec")
            print(f"    Sample shape: {data['sample'].shape}")
            print(f"    Sample stats: mean={np.mean(data['sample']):.4f}, "
                  f"std={np.std(data['sample']):.4f}")

    # ============================================================
    # 7. Final Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("[Final Summary]")
    print(f"{'='*70}")

    checks = {
        "Total samples > 0": len(all_keys) > 0,
        "Valid sample shape": all(s > 0 for s in sample_data['sample'].shape),
        "No NaN values": n_nan == 0,
        "No Inf values": n_inf == 0,
    }

    if is_merged:
        checks["Train samples > 0"] = len(dataset.get('train', [])) > 0
        checks["Val samples > 0"] = len(dataset.get('val', [])) > 0
        checks["Test samples > 0"] = len(dataset.get('test', [])) > 0

    all_passed = True
    print("\nValidation Results:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Dataset is ready for training.")
    else:
        print("⚠️  SOME CHECKS FAILED! Please review the results above.")
    print(f"{'='*70}")

env.close()
