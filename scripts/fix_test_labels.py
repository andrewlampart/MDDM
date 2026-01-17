"""
Fix test set labels - create proper test split with labels.

DAIC-WOZ AVEC2017 test_split CSV doesn't contain PHQ labels.
This script creates a corrected test split using the full labels.

Option 1: If you have access to full DAIC-WOZ labels, update test_split.
Option 2: Use dev set as test, create new val from train.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG


def check_current_splits():
    """Diagnose current split files"""
    print("=" * 60)
    print("Checking current split files")
    print("=" * 60)
    
    splits = {
        'train': CONFIG.DATA_ROOT / "train_split_Depression_AVEC2017.csv",
        'dev': CONFIG.DATA_ROOT / "dev_split_Depression_AVEC2017.csv", 
        'test': CONFIG.DATA_ROOT / "test_split_Depression_AVEC2017.csv",
    }
    
    for name, path in splits.items():
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{name}: {len(df)} samples")
            print(f"  Columns: {list(df.columns)}")
            if 'PHQ8_Binary' in df.columns:
                print(f"  Depression cases: {df['PHQ8_Binary'].sum()} ({df['PHQ8_Binary'].mean():.1%})")
            else:
                print(f"  WARNING: No PHQ8_Binary column!")
        else:
            print(f"\n{name}: FILE NOT FOUND")


def fix_test_split_option2():
    """
    Option 2: Use dev as test, split train into new train+val
    
    Original DAIC-WOZ AVEC2017:
    - Train: 107 samples (30 depressed)
    - Dev: 35 samples (12 depressed)  
    - Test: 47 samples (labels hidden)
    
    New split:
    - Train: ~85 samples from original train
    - Val: ~22 samples from original train
    - Test: 35 samples from original dev (has labels!)
    """
    print("\n" + "=" * 60)
    print("Option 2: Using DEV set as TEST, splitting TRAIN into TRAIN+VAL")
    print("=" * 60)
    
    train_path = CONFIG.DATA_ROOT / "train_split_Depression_AVEC2017.csv"
    dev_path = CONFIG.DATA_ROOT / "dev_split_Depression_AVEC2017.csv"
    
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Train or dev split not found!")
    
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    
    print(f"\nOriginal train: {len(train_df)} samples, {train_df['PHQ8_Binary'].sum()} depressed")
    print(f"Original dev: {len(dev_df)} samples, {dev_df['PHQ8_Binary'].sum()} depressed")
    
    # Dev becomes test (has proper labels)
    new_test_df = dev_df.copy()
    
    # Split original train into new train + val (stratified)
    from sklearn.model_selection import train_test_split
    
    new_train_df, new_val_df = train_test_split(
        train_df,
        test_size=0.2,  # 20% for validation
        stratify=train_df['PHQ8_Binary'],
        random_state=42
    )
    
    print(f"\nNew splits:")
    print(f"  Train: {len(new_train_df)} samples, {new_train_df['PHQ8_Binary'].sum()} depressed ({new_train_df['PHQ8_Binary'].mean():.1%})")
    print(f"  Val: {len(new_val_df)} samples, {new_val_df['PHQ8_Binary'].sum()} depressed ({new_val_df['PHQ8_Binary'].mean():.1%})")
    print(f"  Test: {len(new_test_df)} samples, {new_test_df['PHQ8_Binary'].sum()} depressed ({new_test_df['PHQ8_Binary'].mean():.1%})")
    
    # Save new splits
    backup_dir = CONFIG.DATA_ROOT / "backup_original_splits"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup originals
    import shutil
    for src in [train_path, dev_path]:
        if src.exists():
            shutil.copy(src, backup_dir / src.name)
    print(f"\nOriginal splits backed up to {backup_dir}")
    
    # Save new splits (overwrite)
    new_train_df.to_csv(train_path, index=False)
    new_val_df.to_csv(dev_path, index=False)
    new_test_df.to_csv(CONFIG.DATA_ROOT / "test_split_Depression_AVEC2017.csv", index=False)
    
    print(f"New splits saved!")
    
    return new_train_df, new_val_df, new_test_df


def fix_test_split_option1(full_labels_path: str = None):
    """
    Option 1: Add labels to test split from full DAIC-WOZ labels
    
    Args:
        full_labels_path: Path to CSV with all participant labels
    """
    print("\n" + "=" * 60)
    print("Option 1: Adding labels to test split from full labels file")
    print("=" * 60)
    
    if full_labels_path is None:
        # Try common locations
        possible_paths = [
            CONFIG.DATA_ROOT / "full_test_split.csv",
            CONFIG.DATA_ROOT / "test_labels.csv", 
            CONFIG.DATA_ROOT / "DAIC_WOZ_labels.csv",
        ]
        for p in possible_paths:
            if p.exists():
                full_labels_path = p
                break
    
    if full_labels_path is None or not Path(full_labels_path).exists():
        print("ERROR: Full labels file not found!")
        print("Please provide path to CSV with columns: Participant_ID, PHQ8_Binary, PHQ8_Score")
        print("\nYou can download it from: https://dcapswoz.ict.usc.edu/")
        return None
    
    full_df = pd.read_csv(full_labels_path)
    test_path = CONFIG.DATA_ROOT / "test_split_Depression_AVEC2017.csv"
    test_df = pd.read_csv(test_path)
    
    print(f"Full labels file: {len(full_df)} samples")
    print(f"Columns in full file: {list(full_df.columns)}")
    print(f"Test split file: {len(test_df)} samples")
    print(f"Columns in test file: {list(test_df.columns)}")
    
    # Handle different column name formats
    # full_test_split.csv uses: PHQ_Binary, PHQ_Score
    # train/dev use: PHQ8_Binary, PHQ8_Score
    if 'PHQ_Binary' in full_df.columns:
        full_df = full_df.rename(columns={
            'PHQ_Binary': 'PHQ8_Binary',
            'PHQ_Score': 'PHQ8_Score'
        })
        print("Renamed PHQ_Binary -> PHQ8_Binary, PHQ_Score -> PHQ8_Score")
    
    # Merge labels
    id_col = 'Participant_ID' if 'Participant_ID' in test_df.columns else 'participant_ID'
    
    # Get labels from full file
    label_cols = ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']
    available_cols = [c for c in label_cols if c in full_df.columns]
    
    if 'PHQ8_Binary' not in available_cols:
        print("ERROR: PHQ8_Binary column not found in full labels file!")
        return None
    
    merged = test_df.merge(
        full_df[available_cols],
        left_on=id_col,
        right_on='Participant_ID',
        how='left'
    )
    
    print(f"\nMerged test set: {len(merged)} samples")
    print(f"Depression cases: {merged['PHQ8_Binary'].sum()} ({merged['PHQ8_Binary'].mean():.1%})")
    print(f"Missing labels: {merged['PHQ8_Binary'].isna().sum()}")
    
    # Backup and save
    backup_dir = CONFIG.DATA_ROOT / "backup_original_splits"
    backup_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(test_path, backup_dir / "test_split_original.csv")
    
    merged.to_csv(test_path, index=False)
    print(f"\n✅ Updated test split saved to {test_path}")
    
    return merged


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix test set labels')
    parser.add_argument('--check', action='store_true', help='Check current splits')
    parser.add_argument('--option1', type=str, nargs='?', const='', help='Path to full labels CSV (optional, will search standard locations)')
    parser.add_argument('--option2', action='store_true', help='Use dev as test, split train')
    args = parser.parse_args()
    
    if args.check or (args.option1 is None and not args.option2):
        check_current_splits()
    
    if args.option1 is not None:
        path = args.option1 if args.option1 else None
        fix_test_split_option1(path)
        print("\n" + "=" * 60)
        print("IMPORTANT: Re-run preprocessing pipeline:")
        print("  1. python scripts/01_extract_metadata.py")
        print("  2. python scripts/04_fusion.py")
        print("  3. python scripts/05_train_model.py")
        print("=" * 60)
    
    if args.option2:
        fix_test_split_option2()
        print("\n" + "=" * 60)
        print("IMPORTANT: Re-run preprocessing pipeline:")
        print("  1. python scripts/01_extract_metadata.py")
        print("  2. python scripts/04_fusion.py")
        print("  3. python scripts/05_train_model.py")
        print("=" * 60)
