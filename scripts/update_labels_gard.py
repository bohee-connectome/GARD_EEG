#!/usr/bin/env python3
"""
GARD EEG LMDB Label Updater
- 기존 LMDB에 라벨 정보 추가/업데이트
- preprocessing_gard.py로 생성된 LMDB와 호환
- 새로운 태스크 라벨 추가 시에도 사용

사용법:
    python update_labels_gard.py \
        --lmdb_path /path/to/merged.lmdb \
        --label_path /path/to/labels \
        --dry_run  # 테스트 실행 (실제 변경 안함)

작성일: 2025-01-20
"""

import os
import argparse
import pickle
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
CLASSIFICATION_TASKS = ['task_a', 'task_b', 'task_c', 'task_d', 'task_e']

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='Update labels in existing GARD LMDB')
parser.add_argument('--lmdb_path', type=str, required=True,
                    help='Path to LMDB file to update')
parser.add_argument('--label_path', type=str, required=True,
                    help='Directory containing label CSV files')
parser.add_argument('--tasks', nargs='+', default=CLASSIFICATION_TASKS,
                    help='Tasks to update (default: all)')
parser.add_argument('--dry_run', action='store_true',
                    help='Dry run mode (show changes without applying)')
parser.add_argument('--backup', action='store_true',
                    help='Create backup before modifying')
parser.add_argument('--verbose', action='store_true',
                    help='Show detailed progress')

args = parser.parse_args()


# ============================================================
# Label Loading Functions
# ============================================================
def load_all_labels(label_path, tasks=CLASSIFICATION_TASKS):
    """
    Load label CSV files and create oid -> multi-label mapping

    Args:
        label_path: Directory containing label CSV files
        tasks: List of task names to load

    Returns:
        Dict with oid -> {task_a: label, task_b: label, ...}
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label directory not found: {label_path}")

    oid_to_labels = defaultdict(dict)

    # Task file patterns
    task_files = {
        'task_a': 'task_a_progression_single.csv',
        'task_b': 'task_b_progression_paired.csv',
        'task_c': 'task_c_amyloid.csv',
        'task_d': 'task_d_hippocampus.csv',
        'task_e': 'task_e_mtl_atrophy.csv',
    }

    for task_name in tasks:
        if task_name not in task_files:
            print(f"[Warning] Unknown task: {task_name}")
            continue

        filename = task_files[task_name]
        csv_path = os.path.join(label_path, filename)

        if not os.path.exists(csv_path):
            print(f"[Labels] {task_name}: File not found - {filename}")
            continue

        try:
            df = pd.read_csv(csv_path)
            count = 0
            for _, row in df.iterrows():
                oid = int(row['object_idx'])
                label = row['label']
                if pd.notna(label):
                    # Handle string labels (e.g., 'normal', 'atrophy' in task_d)
                    if isinstance(label, str):
                        label_map = {
                            'normal': 0, 'stable': 0, 'amyloid_negative': 0,
                            'atrophy': 1, 'progressive': 1, 'amyloid_positive': 1
                        }
                        label = label_map.get(label.lower(), None)
                        if label is None:
                            continue
                    oid_to_labels[oid][task_name] = int(label)
                    count += 1
            print(f"[Labels] {task_name}: Loaded {count} labels from {filename}")
        except Exception as e:
            print(f"[Labels] {task_name}: Error loading {filename} - {e}")

    print(f"[Labels] Total unique subjects with labels: {len(oid_to_labels)}")
    return dict(oid_to_labels)


def extract_oid_from_entry(data):
    """
    Extract oid from LMDB entry

    ID Mapping Reference:
        - data_info['oid']: Direct oid if available (new format)
        - data_info['subject_id']: "{year}_{oid}" format (e.g., "2019_23")
        - Fallback: Extract from subject_id string

    Args:
        data: LMDB entry dict

    Returns:
        oid as integer, or None if not extractable
    """
    data_info = data.get('data_info', {})

    # Try direct oid first (new format)
    if 'oid' in data_info and data_info['oid'] is not None:
        return int(data_info['oid'])

    # Try subject_id parsing (format: "2019_23")
    subject_id = data_info.get('subject_id', '')
    if '_' in str(subject_id):
        parts = str(subject_id).split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass

    return None


# ============================================================
# LMDB Update Functions
# ============================================================
def backup_lmdb(lmdb_path):
    """Create backup of LMDB before modification"""
    import shutil
    backup_path = lmdb_path + '.backup'
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(lmdb_path, backup_path)
    print(f"[Backup] Created backup at {backup_path}")
    return backup_path


def analyze_lmdb(lmdb_path, oid_to_labels, tasks):
    """
    Analyze LMDB and show what would be updated

    Args:
        lmdb_path: Path to LMDB
        oid_to_labels: Label mapping
        tasks: Tasks to update

    Returns:
        Dict with statistics
    """
    stats = {
        'total_entries': 0,
        'entries_with_oid': 0,
        'entries_without_oid': 0,
        'labels_found': {task: 0 for task in tasks},
        'labels_missing': {task: 0 for task in tasks},
        'new_labels': {task: 0 for task in tasks},
        'existing_labels': {task: 0 for task in tasks},
        'sample_entries': []
    }

    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b'__keys__':
                continue

            stats['total_entries'] += 1
            data = pickle.loads(value)

            oid = extract_oid_from_entry(data)

            if oid is not None:
                stats['entries_with_oid'] += 1
                subject_labels = oid_to_labels.get(oid, {})

                for task in tasks:
                    # Check if label exists in CSV
                    if task in subject_labels:
                        stats['labels_found'][task] += 1
                        # Check if already has label in LMDB
                        current_label = data.get(task)
                        if current_label is None:
                            stats['new_labels'][task] += 1
                        else:
                            stats['existing_labels'][task] += 1
                    else:
                        stats['labels_missing'][task] += 1

                # Collect sample entries
                if len(stats['sample_entries']) < 3:
                    stats['sample_entries'].append({
                        'key': key.decode(),
                        'oid': oid,
                        'labels': subject_labels
                    })
            else:
                stats['entries_without_oid'] += 1

    env.close()
    return stats


def update_lmdb_labels(lmdb_path, oid_to_labels, tasks, dry_run=False, verbose=False):
    """
    Update labels in LMDB

    Args:
        lmdb_path: Path to LMDB
        oid_to_labels: Label mapping
        tasks: Tasks to update
        dry_run: If True, don't actually modify
        verbose: Show detailed progress

    Returns:
        Dict with update statistics
    """
    stats = {
        'processed': 0,
        'updated': 0,
        'skipped_no_oid': 0,
        'skipped_no_label': 0,
        'task_updates': {task: 0 for task in tasks}
    }

    if dry_run:
        print("\n[DRY RUN] No changes will be made\n")

    # Open in read-write mode
    env = lmdb.open(lmdb_path, readonly=dry_run, lock=not dry_run,
                    map_size=30*1024*1024*1024)  # 30GB

    # First pass: collect all keys (can't modify while iterating)
    keys_to_update = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b'__keys__':
                continue
            keys_to_update.append(key)

    # Second pass: update entries
    for key in tqdm(keys_to_update, desc="Updating labels", disable=not verbose):
        with env.begin(write=not dry_run) as txn:
            value = txn.get(key)
            if value is None:
                continue

            data = pickle.loads(value)
            oid = extract_oid_from_entry(data)

            stats['processed'] += 1

            if oid is None:
                stats['skipped_no_oid'] += 1
                continue

            subject_labels = oid_to_labels.get(oid, {})

            if not subject_labels:
                stats['skipped_no_label'] += 1
                continue

            # Update labels
            modified = False
            for task in tasks:
                if task in subject_labels:
                    new_label = subject_labels[task]
                    old_label = data.get(task)

                    if old_label != new_label:
                        data[task] = new_label
                        stats['task_updates'][task] += 1
                        modified = True

                        if verbose:
                            print(f"  {key.decode()}: {task} {old_label} -> {new_label}")

            if modified:
                stats['updated'] += 1
                if not dry_run:
                    txn.put(key, pickle.dumps(data))

    env.close()
    return stats


def print_stats(stats, title="Statistics"):
    """Pretty print statistics"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    if 'total_entries' in stats:
        print(f"\n📊 LMDB Analysis:")
        print(f"   Total entries:      {stats['total_entries']:,}")
        print(f"   With oid:           {stats['entries_with_oid']:,}")
        print(f"   Without oid:        {stats['entries_without_oid']:,}")

        print(f"\n📋 Label Coverage:")
        for task in stats['labels_found']:
            found = stats['labels_found'][task]
            new = stats['new_labels'][task]
            existing = stats['existing_labels'][task]
            missing = stats['labels_missing'][task]
            print(f"   {task}: found={found}, new={new}, existing={existing}, missing={missing}")

        if stats['sample_entries']:
            print(f"\n📝 Sample Entries:")
            for entry in stats['sample_entries']:
                print(f"   {entry['key']}")
                print(f"      oid: {entry['oid']}, labels: {entry['labels']}")

    if 'processed' in stats:
        print(f"\n📈 Update Results:")
        print(f"   Processed:          {stats['processed']:,}")
        print(f"   Updated:            {stats['updated']:,}")
        print(f"   Skipped (no oid):   {stats['skipped_no_oid']:,}")
        print(f"   Skipped (no label): {stats['skipped_no_label']:,}")

        print(f"\n📋 Task Updates:")
        for task, count in stats['task_updates'].items():
            print(f"   {task}: {count:,} updates")


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("GARD EEG LMDB Label Updater")
    print("="*60)
    print(f"LMDB path:   {args.lmdb_path}")
    print(f"Label path:  {args.label_path}")
    print(f"Tasks:       {args.tasks}")
    print(f"Dry run:     {args.dry_run}")
    print(f"Backup:      {args.backup}")

    # Validate paths
    if not os.path.exists(args.lmdb_path):
        print(f"[Error] LMDB not found: {args.lmdb_path}")
        return

    # Load labels
    print(f"\n{'─'*40}")
    print("Loading labels...")
    oid_to_labels = load_all_labels(args.label_path, args.tasks)

    if not oid_to_labels:
        print("[Error] No labels loaded. Check label_path and CSV files.")
        return

    # Analyze LMDB
    print(f"\n{'─'*40}")
    print("Analyzing LMDB...")
    analysis_stats = analyze_lmdb(args.lmdb_path, oid_to_labels, args.tasks)
    print_stats(analysis_stats, "LMDB Analysis")

    # Confirm before update (unless dry_run)
    if not args.dry_run:
        print(f"\n{'─'*40}")
        confirm = input("Proceed with update? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

        # Backup if requested
        if args.backup:
            backup_lmdb(args.lmdb_path)

    # Update labels
    print(f"\n{'─'*40}")
    print("Updating labels...")
    update_stats = update_lmdb_labels(
        args.lmdb_path,
        oid_to_labels,
        args.tasks,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    print_stats(update_stats, "Update Results")

    print(f"\n{'='*60}")
    if args.dry_run:
        print("🔍 DRY RUN completed. No changes were made.")
        print("   Remove --dry_run to apply changes.")
    else:
        print("✅ Label update completed!")
    print("="*60)


if __name__ == '__main__':
    main()
