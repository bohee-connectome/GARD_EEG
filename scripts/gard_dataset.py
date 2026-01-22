#!/usr/bin/env python3
"""
GARD Custom DataLoader for DIVER Finetuning

멀티태스크 구조 (task_a ~ task_e) 지원
- label 필드 대신 target_task에서 읽음
- None인 entry 자동 필터링

LMDB 경로 (Perlmutter):
    /pscratch/sd/b/boheelee/GARD/EEG/lmdb/
    ├── attention/merged_resample-500_highpass-0.3_lowpass-200.lmdb
    ├── beam/merged_resample-500_highpass-0.3_lowpass-200.lmdb
    └── sensory/merged_resample-500_highpass-0.3_lowpass-200.lmdb

Task 정의:
    task_a: Progression (Single) - CN→MCI 진행 예측 (16%)
    task_b: Progression (Paired) - 두 시점 EEG 변화 (11%)
    task_c: Amyloid - Aβ+ vs Aβ- 예측 (34%)
    task_d: Hippocampus - 해마 위축 예측 (8%)
    task_e: MTL Atrophy - MTL 피질 위축 예측 (5%)

Usage:
    # Task A (Progression Single) 학습
    train_set = GARDCustomDataset(
        '/pscratch/sd/b/boheelee/GARD/EEG/lmdb/attention/merged_...',
        mode='train',
        target_task='task_a'
    )

작성일: 2026-01-22
"""

import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class GARDCustomDataset(Dataset):
    """GARD 멀티태스크용 Custom Dataset"""

    def __init__(self, datasets_dir, mode='train', target_task='task_a',
                 transform=None, return_info=True, collate_fn=None):
        """
        Args:
            datasets_dir: LMDB 경로
            mode: 'train', 'val', 'test'
            target_task: 'task_a' | 'task_b' | 'task_c' | 'task_d' | 'task_e'
            transform: 데이터 변환
            return_info: data_info 반환 여부 (기본 True - finetuning에서 필요)
            collate_fn: DataLoader용 collate 함수
        """
        self.db = lmdb.open(datasets_dir, readonly=True, lock=False,
                           readahead=True, meminit=False)
        self.mode = mode
        self.target_task = target_task
        self.transform = transform
        self.return_info = return_info
        self.collate = collate_fn

        # 해당 task의 label이 None이 아닌 key만 필터링
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))[self.mode]

            self.keys = []
            label_counts = {0: 0, 1: 0}
            for key in all_keys:
                data = pickle.loads(txn.get(key.encode()))
                label = data.get(self.target_task)
                if label is not None:
                    self.keys.append(key)
                    label_counts[label] = label_counts.get(label, 0) + 1

        # 통계 출력
        total_valid = len(self.keys)
        ratio = label_counts.get(1, 0) / total_valid if total_valid > 0 else 0
        print(f"[GARD] {mode}: {total_valid}/{len(all_keys)} samples "
              f"({target_task}, pos_ratio={ratio:.1%})")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            data = pickle.loads(txn.get(key.encode()))

        sample = data['sample']
        label = data[self.target_task]  # 핵심: task_X에서 label 읽음
        data_info = data.get('data_info', {})

        # sample_key 추가 (디버깅용)
        data_info['sample_key'] = key
        data_info['target_task'] = self.target_task

        sample = torch.from_numpy(sample).float()
        if self.transform:
            sample = self.transform(sample)

        if self.return_info:
            return sample, label, data_info
        return sample, label


class GARDLoadDataset:
    """GARD DataLoader Factory - generalized_datasets.py 호환"""

    def __init__(self, dataset_class, params):
        """
        Args:
            dataset_class: GARDCustomDataset 클래스
            params: argparse namespace (datasets_dir, batch_size, gard_task 등)
        """
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.dataset_class = dataset_class
        self.target_task = getattr(params, 'gard_task', 'task_a')

    def prepare_datasets(self):
        """Train, Val, Test 데이터셋 준비"""
        # collate_fn import (DIVER 구조)
        try:
            from datasets.dataloader_utils import collate_fn_for_data_info_finetuning
            collate_fn = collate_fn_for_data_info_finetuning
        except ImportError:
            # Fallback: 기본 collate
            collate_fn = None
            print("[GARD] Warning: Using default collate_fn")

        transform = None
        if hasattr(self.params, 'ablation') and self.params.ablation == "ch_permute_finetune":
            from utils.util import ChannelPermutation
            transform = ChannelPermutation()

        train_set = self.dataset_class(
            self.datasets_dir,
            mode='train',
            target_task=self.target_task,
            transform=transform,
            collate_fn=collate_fn
        )

        val_set = self.dataset_class(
            self.datasets_dir,
            mode='val',
            target_task=self.target_task,
            transform=transform,
            collate_fn=collate_fn
        )

        test_set = self.dataset_class(
            self.datasets_dir,
            mode='test',
            target_task=self.target_task,
            collate_fn=collate_fn  # test에는 transform 없음
        )

        return train_set, val_set, test_set

    def get_data_loader(self):
        """DataLoader 생성"""
        train_set, val_set, test_set = self.prepare_datasets()

        print(f"\n[GARD] Dataset Summary:")
        print(f"  Train: {len(train_set)}")
        print(f"  Val:   {len(val_set)}")
        print(f"  Test:  {len(test_set)}")
        print(f"  Total: {len(train_set) + len(val_set) + len(test_set)}")

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=getattr(self.params, 'num_workers', 0)
            ),
        }

        return data_loader


def get_gard_loader(params):
    """
    GARD DataLoader 팩토리 함수

    Usage in generalized_datasets.py:
        # datasets dict에 추가:
        'gard': (GARDLoadDataset, GARDCustomDataset),

        # 또는 직접 호출:
        if dataset_name == 'gard':
            from gard_dataset import get_gard_loader
            return get_gard_loader(params)
    """
    return GARDLoadDataset(GARDCustomDataset, params)


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test GARD Dataset')
    parser.add_argument('--lmdb_path', type=str, required=True,
                       help='Path to GARD LMDB')
    parser.add_argument('--task', type=str, default='task_a',
                       choices=['task_a', 'task_b', 'task_c', 'task_d', 'task_e'],
                       help='Target classification task')
    args = parser.parse_args()

    print("=" * 60)
    print("GARD Dataset Test")
    print("=" * 60)
    print(f"LMDB: {args.lmdb_path}")
    print(f"Task: {args.task}")
    print()

    # 각 split 테스트
    for mode in ['train', 'val', 'test']:
        dataset = GARDCustomDataset(
            args.lmdb_path,
            mode=mode,
            target_task=args.task
        )
        print(f"  {mode}: {len(dataset)} samples")

        # 첫 샘플 확인
        if len(dataset) > 0:
            sample, label, info = dataset[0]
            print(f"    Sample shape: {sample.shape}")
            print(f"    Label: {label}")
            print(f"    Subject: {info.get('subject_id', 'N/A')}")

    print("\n✅ Test completed!")
