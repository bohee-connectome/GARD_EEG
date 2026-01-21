#!/usr/bin/env python3
"""
Add 'label' field to GARD LMDB for DIVER compatibility.

This script copies a specific task label (task_a ~ task_e) to the 'label' field
so that DIVER's standard DataLoader can read it.

Usage:
    python add_label_field.py --lmdb_path /path/to/lmdb --task task_c
    python add_label_field.py --lmdb_path /path/to/lmdb --task task_c --dry_run
"""

import os
import argparse
import pickle
import lmdb
from tqdm import tqdm


# Task mapping for reference
TASK_NAMES = {
    'task_a': 'Progression (Single)',
    'task_b': 'Progression (Paired)',
    'task_c': 'Amyloid Prediction',
    'task_d': 'Hippocampus Atrophy',
    'task_e': 'MTL Cortical Atrophy',
}


def analyze_lmdb(lmdb_path, task):
    """Analyze LMDB to show statistics before update."""
    stats = {
        'total_entries': 0,
        'has_task_label': 0,
        'task_label_none': 0,
        'task_label_0': 0,
        'task_label_1': 0,
        'already_has_label': 0,
        'sample_entries': [],
    }

    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b'__keys__':
                continue

            stats['total_entries'] += 1
            data = pickle.loads(value)

            # Check task label
            task_label = data.get(task)
            if task_label is not None:
                stats['has_task_label'] += 1
                if task_label == 0:
                    stats['task_label_0'] += 1
                elif task_label == 1:
                    stats['task_label_1'] += 1
            else:
                stats['task_label_none'] += 1

            # Check if already has 'label' field
            if 'label' in data:
                stats['already_has_label'] += 1

            # Collect sample entries
            if len(stats['sample_entries']) < 3 and task_label is not None:
                stats['sample_entries'].append({
                    'key': key.decode(),
                    'task_label': task_label,
                    'has_label': 'label' in data,
                })

    env.close()
    return stats


def add_label_field(lmdb_path, task, dry_run=False, verbose=False):
    """
    Add 'label' field to LMDB entries.

    Args:
        lmdb_path: Path to LMDB
        task: Task to use as label source (task_a, task_b, etc.)
        dry_run: If True, don't actually modify
        verbose: Show detailed progress

    Returns:
        Dict with update statistics
    """
    stats = {
        'processed': 0,
        'updated': 0,
        'skipped_none': 0,
        'already_set': 0,
    }

    if dry_run:
        print("\n[DRY RUN] No changes will be made\n")

    # Open in read-write mode
    env = lmdb.open(lmdb_path, readonly=dry_run, lock=not dry_run,
                    map_size=30*1024*1024*1024)  # 30GB

    # First pass: collect all keys
    keys_to_update = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b'__keys__':
                continue
            keys_to_update.append(key)

    # Second pass: update entries
    for key in tqdm(keys_to_update, desc="Adding label field", disable=not verbose):
        with env.begin(write=not dry_run) as txn:
            value = txn.get(key)
            if value is None:
                continue

            data = pickle.loads(value)
            stats['processed'] += 1

            # Get task label
            task_label = data.get(task)

            # Skip if task label is None
            if task_label is None:
                stats['skipped_none'] += 1
                continue

            # Check if label already set to same value
            current_label = data.get('label')
            if current_label == task_label:
                stats['already_set'] += 1
                continue

            # Add/update label field
            data['label'] = task_label
            stats['updated'] += 1

            if verbose:
                print(f"  {key.decode()}: label = {task_label}")

            if not dry_run:
                txn.put(key, pickle.dumps(data))

    env.close()
    return stats


def print_stats(stats, title="Statistics"):
    """Pretty print statistics."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    if 'total_entries' in stats:
        print(f"\n📊 LMDB Analysis:")
        print(f"   Total entries:        {stats['total_entries']:,}")
        print(f"   Has task label:       {stats['has_task_label']:,}")
        print(f"   Task label = 0:       {stats['task_label_0']:,}")
        print(f"   Task label = 1:       {stats['task_label_1']:,}")
        print(f"   Task label = None:    {stats['task_label_none']:,}")
        print(f"   Already has 'label':  {stats['already_has_label']:,}")

        if stats['sample_entries']:
            print(f"\n📝 Sample Entries:")
            for entry in stats['sample_entries']:
                print(f"   {entry['key']}")
                print(f"      task_label: {entry['task_label']}, has_label: {entry['has_label']}")

    if 'processed' in stats:
        print(f"\n📈 Update Results:")
        print(f"   Processed:     {stats['processed']:,}")
        print(f"   Updated:       {stats['updated']:,}")
        print(f"   Skipped (None):{stats['skipped_none']:,}")
        print(f"   Already set:   {stats['already_set']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Add 'label' field to GARD LMDB for DIVER compatibility"
    )
    parser.add_argument('--lmdb_path', required=True, help='Path to LMDB directory')
    parser.add_argument('--task', required=True, choices=['task_a', 'task_b', 'task_c', 'task_d', 'task_e'],
                        help='Task to use as label source')
    parser.add_argument('--dry_run', action='store_true', help='Analyze only, do not modify')
    parser.add_argument('--backup', action='store_true', help='Create backup before modification')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')

    args = parser.parse_args()

    print("="*60)
    print("GARD LMDB - Add Label Field")
    print("="*60)
    print(f"LMDB path: {args.lmdb_path}")
    print(f"Task:      {args.task} ({TASK_NAMES.get(args.task, 'Unknown')})")
    print(f"Dry run:   {args.dry_run}")
    print(f"Backup:    {args.backup}")

    # Validate path
    if not os.path.exists(args.lmdb_path):
        print(f"[Error] LMDB not found: {args.lmdb_path}")
        return

    # Analyze first
    print(f"\n{'─'*40}")
    print("Analyzing LMDB...")
    analysis_stats = analyze_lmdb(args.lmdb_path, args.task)
    print_stats(analysis_stats, f"LMDB Analysis ({args.task})")

    # Confirm before update
    if not args.dry_run:
        print(f"\n{'─'*40}")
        print(f"Will add 'label' field from '{args.task}' to {analysis_stats['has_task_label']:,} entries")
        confirm = input("Proceed? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

        # Backup if requested
        if args.backup:
            import shutil
            backup_path = args.lmdb_path + '.backup_label'
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(args.lmdb_path, backup_path)
            print(f"[Backup] Created backup at {backup_path}")

    # Add label field
    print(f"\n{'─'*40}")
    print("Adding label field...")
    update_stats = add_label_field(args.lmdb_path, args.task,
                                    dry_run=args.dry_run, verbose=args.verbose)
    print_stats(update_stats, "Update Results")

    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN completed. No changes made.")
    else:
        print(f"Done! Added 'label' field to {update_stats['updated']:,} entries.")
    print("="*60)


if __name__ == '__main__':
    main()
