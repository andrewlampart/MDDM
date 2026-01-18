#!/usr/bin/env python
"""
Ablation Study Script - Systematyczne testowanie modalności

Ten skrypt uruchamia pełne ablation study na danych DAIC-WOZ,
testując wszystkie kombinacje modalności (audio, text, demographic).

Użycie:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --output results/ablation
    python scripts/ablation_study.py --n-splits 10

Wymagania:
    - Przetworzone features w data/processed/
    - XGBoost zainstalowany
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from evaluation.ablation import AblationStudy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features(data_dir: Path) -> dict:
    """
    Load all processed features from data directory.
    
    Expected files:
    - audio_features.csv or audio_features.npy
    - text_features.csv or text_features.npy
    - labels.csv or labels.npy (or daic_labels.csv)
    
    Returns:
        dict with X_audio, X_text, y, and optionally X_demo
    """
    data = {}
    
    # Load audio features
    audio_path = data_dir / 'audio_features.csv'
    if audio_path.exists():
        audio_df = pd.read_csv(audio_path)
        # Remove non-feature columns
        feature_cols = [c for c in audio_df.columns if c not in ['participant_id', 'session_id', 'label']]
        data['X_audio'] = audio_df[feature_cols].values
        logger.info(f"Loaded audio features: {data['X_audio'].shape}")
    else:
        npy_path = data_dir / 'audio_features.npy'
        if npy_path.exists():
            data['X_audio'] = np.load(npy_path)
            logger.info(f"Loaded audio features: {data['X_audio'].shape}")
        else:
            logger.warning("No audio features found!")
            return None
    
    # Load text features
    text_path = data_dir / 'text_features.csv'
    if text_path.exists():
        text_df = pd.read_csv(text_path)
        feature_cols = [c for c in text_df.columns if c not in ['participant_id', 'session_id', 'label']]
        data['X_text'] = text_df[feature_cols].values
        logger.info(f"Loaded text features: {data['X_text'].shape}")
    else:
        npy_path = data_dir / 'text_features.npy'
        if npy_path.exists():
            data['X_text'] = np.load(npy_path)
            logger.info(f"Loaded text features: {data['X_text'].shape}")
        else:
            logger.warning("No text features found!")
            return None
    
    # Load labels
    labels_path = data_dir / 'daic_labels.csv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        if 'PHQ8_Binary' in labels_df.columns:
            data['y'] = labels_df['PHQ8_Binary'].values
        elif 'label' in labels_df.columns:
            data['y'] = labels_df['label'].values
        else:
            data['y'] = labels_df.iloc[:, -1].values
        logger.info(f"Loaded labels: {len(data['y'])} samples, {data['y'].mean():.1%} positive")
    else:
        npy_path = data_dir / 'y_train.npy'  # Alternative
        if npy_path.exists():
            data['y'] = np.load(npy_path)
        else:
            logger.warning("No labels found!")
            return None
    
    # Load demographic features (optional)
    demo_path = data_dir / 'demographic_features.csv'
    if demo_path.exists():
        demo_df = pd.read_csv(demo_path)
        feature_cols = [c for c in demo_df.columns if c not in ['participant_id', 'session_id', 'label']]
        data['X_demo'] = demo_df[feature_cols].values
        logger.info(f"Loaded demographic features: {data['X_demo'].shape}")
    
    return data


def get_model_factory(model_type: str = 'xgboost'):
    """
    Get model factory function for ablation study.
    
    Args:
        model_type: 'xgboost', 'random_forest', or 'logistic'
        
    Returns:
        Callable that returns fresh model instance
    """
    if model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            
            def factory():
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=2.3,  # For class imbalance
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            return factory
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")
            model_type = 'random_forest'
    
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        
        def factory():
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        return factory
    
    if model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        
        def factory():
            return LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        return factory
    
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study on MDDM features'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('data/processed'),
        help='Directory with processed features'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results/ablation'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-splits', '-n',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['xgboost', 'random_forest', 'logistic'],
        default='xgboost',
        help='Model type to use'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("MDDM ABLATION STUDY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("\nMake sure to run the feature extraction pipeline first:")
        print("  python scripts/02_extract_audio_features.py")
        print("  python scripts/03_extract_text_features.py")
        return 1
    
    print(f"\nLoading features from: {data_dir}")
    data = load_features(data_dir)
    
    if data is None:
        print("\n[ERROR] Failed to load features!")
        print("Required files:")
        print("  - audio_features.csv or audio_features.npy")
        print("  - text_features.csv or text_features.npy")
        print("  - daic_labels.csv or labels.npy")
        return 1
    
    # Validate shapes
    n_audio = len(data['X_audio'])
    n_text = len(data['X_text'])
    n_labels = len(data['y'])
    
    if not (n_audio == n_text == n_labels):
        print(f"\n[ERROR] Sample count mismatch!")
        print(f"  Audio: {n_audio}")
        print(f"  Text: {n_text}")
        print(f"  Labels: {n_labels}")
        return 1
    
    print(f"\nDataset summary:")
    print(f"  Samples: {n_labels}")
    print(f"  Audio features: {data['X_audio'].shape[1]}")
    print(f"  Text features: {data['X_text'].shape[1]}")
    if 'X_demo' in data:
        print(f"  Demographic features: {data['X_demo'].shape[1]}")
    print(f"  Class balance: {data['y'].mean():.1%} positive")
    
    # Get model factory
    print(f"\nModel: {args.model}")
    model_factory = get_model_factory(args.model)
    
    # Create ablation study
    ablation = AblationStudy(
        X_audio=data['X_audio'],
        X_text=data['X_text'],
        y=data['y'],
        X_demographic=data.get('X_demo')
    )
    
    # Run ablation
    print(f"\nRunning {args.n_splits}-fold cross-validation...")
    results = ablation.run_full_ablation(
        model_factory=model_factory,
        n_splits=args.n_splits,
        random_state=args.seed,
        verbose=True
    )
    
    # Print report
    print("\n" + ablation.generate_report())
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    ablation.save_results(output_dir)
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'ablation_results.csv'}")
    print(f"  - {output_dir / 'ablation_detailed.json'}")
    print(f"  - {output_dir / 'synergy_analysis.json'}")
    print(f"  - {output_dir / 'ablation_plot.png'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
