"""
MIL DataLoader for Depression Detection

Provides PyTorch Dataset and DataLoader for Multi-Instance Learning.
Each sample is a "bag" (complete interview) with variable number of instances.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MIL DataLoader disabled.")


if TORCH_AVAILABLE:
    
    class MILBagDataset(Dataset):
        """
        Dataset for Multi-Instance Learning.
        
        Returns complete "bags" (interviews) where each bag contains
        variable number of instances (patient responses).
        
        The bag-level label is the depression diagnosis.
        """
        
        def __init__(self, 
                     instances_csv: Path = None,
                     session_ids: List[int] = None,
                     transform=None):
            """
            Args:
                instances_csv: Path to instances CSV
                session_ids: List of session IDs to include (for train/val/test split)
                transform: Optional transform to apply to texts
            """
            instances_csv = instances_csv or CONFIG.INSTANCES_CSV
            
            if not Path(instances_csv).exists():
                raise FileNotFoundError(
                    f"Instances file not found: {instances_csv}\n"
                    f"Run: python data/interview_segmentation.py"
                )
            
            self.df = pd.read_csv(instances_csv)
            self.transform = transform
            
            # Filter by session_ids if provided
            if session_ids is not None:
                self.df = self.df[self.df['session_id'].isin(session_ids)]
            
            # Group by session to create bags
            self.bags = []
            for session_id, group in self.df.groupby('session_id'):
                # Sort by sequence number
                group = group.sort_values('sequence_num')
                
                instances = group['text'].tolist()
                label = group['depression'].iloc[0]
                
                self.bags.append({
                    'session_id': session_id,
                    'instances': instances,
                    'label': int(label),
                    'n_instances': len(instances)
                })
            
            print(f"Created dataset with {len(self.bags)} bags")
        
        def __len__(self):
            return len(self.bags)
        
        def __getitem__(self, idx):
            bag = self.bags[idx]
            
            instances = bag['instances']
            if self.transform:
                instances = [self.transform(inst) for inst in instances]
            
            return {
                'bag_instances': instances,
                'label': torch.tensor(bag['label'], dtype=torch.float),
                'session_id': bag['session_id'],
                'n_instances': bag['n_instances']
            }
    
    
    def mil_collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for MIL batches.
        
        Handles variable-length bags by keeping instances as lists.
        
        Args:
            batch: List of samples from MILBagDataset
            
        Returns:
            Batched dictionary
        """
        return {
            'bag_instances': [item['bag_instances'] for item in batch],
            'labels': torch.stack([item['label'] for item in batch]),
            'session_ids': [item['session_id'] for item in batch],
            'n_instances': [item['n_instances'] for item in batch]
        }
    
    
    def create_mil_dataloaders(
        instances_csv: Path = None,
        batch_size: int = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test DataLoaders with stratified split.
        
        Args:
            instances_csv: Path to instances CSV
            batch_size: Batch size
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        instances_csv = instances_csv or CONFIG.INSTANCES_CSV
        batch_size = batch_size or CONFIG.MIL_BATCH_SIZE
        random_state = random_state or CONFIG.RANDOM_SEED
        
        # Load instances to get session IDs and labels
        df = pd.read_csv(instances_csv)
        
        # Get unique sessions with their labels
        sessions = df.groupby('session_id')['depression'].first().reset_index()
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        np.random.seed(random_state)
        
        # First split: train+val vs test
        test_ratio = 1 - train_ratio - val_ratio
        trainval_sessions, test_sessions = train_test_split(
            sessions['session_id'].values,
            test_size=test_ratio,
            stratify=sessions['depression'].values,
            random_state=random_state
        )
        
        # Get labels for trainval
        trainval_labels = sessions[sessions['session_id'].isin(trainval_sessions)]['depression'].values
        
        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_sessions, val_sessions = train_test_split(
            trainval_sessions,
            test_size=val_ratio_adjusted,
            stratify=trainval_labels,
            random_state=random_state
        )
        
        print(f"Split: Train={len(train_sessions)}, Val={len(val_sessions)}, Test={len(test_sessions)}")
        
        # Create datasets
        train_dataset = MILBagDataset(instances_csv, session_ids=train_sessions.tolist())
        val_dataset = MILBagDataset(instances_csv, session_ids=val_sessions.tolist())
        test_dataset = MILBagDataset(instances_csv, session_ids=test_sessions.tolist())
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    
    def create_mil_dataloaders_from_splits(
        instances_csv: Path = None,
        labels_csv: Path = None,
        batch_size: int = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders using existing train/val/test session splits.
        
        Uses DAIC-WOZ standard splits from dev/test CSV files.
        
        Args:
            instances_csv: Path to instances CSV
            labels_csv: Path to labels CSV (with train/val/test info)
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        instances_csv = instances_csv or CONFIG.INSTANCES_CSV
        labels_csv = labels_csv or CONFIG.LABELS_CSV
        batch_size = batch_size or CONFIG.MIL_BATCH_SIZE
        
        # Load original split files
        train_split_path = CONFIG.DATA_ROOT / "train_split_Depression_AVEC2017.csv"
        dev_split_path = CONFIG.DATA_ROOT / "dev_split_Depression_AVEC2017.csv"
        test_split_path = CONFIG.DATA_ROOT / "test_split_Depression_AVEC2017.csv"
        
        train_sessions = []
        val_sessions = []
        test_sessions = []
        
        # Get session IDs from split files
        for split_path, session_list in [
            (train_split_path, train_sessions),
            (dev_split_path, val_sessions),
            (test_split_path, test_sessions)
        ]:
            if split_path.exists():
                df = pd.read_csv(split_path)
                id_col = 'Participant_ID' if 'Participant_ID' in df.columns else 'participant_ID'
                if id_col in df.columns:
                    session_list.extend(df[id_col].tolist())
        
        if not train_sessions:
            print("Standard splits not found, using random split")
            return create_mil_dataloaders(instances_csv, batch_size)
        
        # Filter to sessions that have instances
        df_instances = pd.read_csv(instances_csv)
        available_sessions = set(df_instances['session_id'].unique())
        
        train_sessions = [s for s in train_sessions if s in available_sessions]
        val_sessions = [s for s in val_sessions if s in available_sessions]
        test_sessions = [s for s in test_sessions if s in available_sessions]
        
        print(f"Using DAIC-WOZ splits: Train={len(train_sessions)}, Val={len(val_sessions)}, Test={len(test_sessions)}")
        
        # Create datasets
        train_dataset = MILBagDataset(instances_csv, session_ids=train_sessions)
        val_dataset = MILBagDataset(instances_csv, session_ids=val_sessions)
        test_dataset = MILBagDataset(instances_csv, session_ids=test_sessions)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch required for MIL DataLoader")
    else:
        print("Testing MIL DataLoader...")
        
        try:
            train_loader, val_loader, test_loader = create_mil_dataloaders()
            
            # Test iteration
            for batch in train_loader:
                print(f"\nBatch:")
                print(f"  Bags: {len(batch['bag_instances'])}")
                print(f"  Labels: {batch['labels']}")
                print(f"  Instances per bag: {batch['n_instances']}")
                print(f"  Sample text: {batch['bag_instances'][0][0][:100]}...")
                break
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run interview_segmentation.py first to create instances.")
