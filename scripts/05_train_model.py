"""Step 5: Train baseline model and evaluate"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline import BaselineClassifier
from src.utils.config import Config


def main():
    """Train baseline model with feature selection and threshold optimization"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-search', action='store_true', help='Run grid search')
    parser.add_argument('--n-features', type=int, default=50, help='Number of features to select')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 5: Train Baseline Model (Improved)")
    print("=" * 60)
    
    Config.create_directories()
    
    # Load fused features
    csv_path = Config.FUSED_FEATURES_H5.with_suffix('.csv')
    
    if csv_path.exists():
        print(f"Loading fused features from CSV...")
        df = pd.read_csv(csv_path)
    elif Config.FUSED_FEATURES_H5.exists():
        print(f"Loading fused features from HDF5...")
        df = pd.read_hdf(Config.FUSED_FEATURES_H5, key='data')
    else:
        raise FileNotFoundError(
            f"Fused features not found: {csv_path} or {Config.FUSED_FEATURES_H5}\n"
            f"Run 04_fusion.py first."
        )
    
    print(f"Loaded {len(df)} samples with {len(df.columns)-2} features")
    
    # Initialize classifier
    classifier = BaselineClassifier(
        n_estimators=100,
        max_depth=10,  # Ograniczenie głębokości
        random_state=Config.RANDOM_SEED,
        n_features=args.n_features
    )
    
    # Split data
    print("\nSplitting data into train/val/test...")
    train_df, val_df, test_df = classifier.train_test_split_stratified(
        df,
        train_size=Config.TRAIN_SIZE,
        val_size=Config.VAL_SIZE,
        test_size=Config.TEST_SIZE
    )
    
    print(f"Train: {len(train_df)} samples ({train_df['depression'].sum()} positive)")
    print(f"Val:   {len(val_df)} samples ({val_df['depression'].sum()} positive)")
    print(f"Test:  {len(test_df)} samples ({test_df['depression'].sum()} positive)")
    
    # Prepare data
    X_train, y_train = classifier.prepare_data(train_df)
    X_val, y_val = classifier.prepare_data(val_df)
    X_test, y_test = classifier.prepare_data(test_df)
    
    print(f"\nOriginal feature dimensions: {X_train.shape[1]}")
    
    # Train - z opcjonalnym grid search
    if args.grid_search:
        print("\n" + "-" * 40)
        print("Running Grid Search for optimal hyperparameters...")
        print("-" * 40)
        best_params = classifier.grid_search(X_train, y_train)
        # Po grid search optymalizuj próg
        classifier._optimize_threshold(X_val, y_val)
    else:
        print("\nTraining model with feature selection...")
        classifier.fit(X_train, y_train, X_val=X_val, y_val=y_val, optimize_threshold=True)
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluating on validation set...")
    val_metrics = classifier.evaluate(X_val, y_val, set_name="Validation")
    
    print("\n" + "-" * 40)
    print("Evaluating on test set...")
    test_metrics = classifier.evaluate(X_test, y_test, set_name="Test")
    
    # Feature importance
    print("\n" + "-" * 40)
    print("Top 10 most important features:")
    if hasattr(classifier.model, 'feature_importances_'):
        importances = classifier.model.feature_importances_
        feature_cols = [col for col in train_df.columns 
                       if col not in ['session_id', 'depression', 'phq8_score', 'qids_score']]
        if classifier.selected_features is not None:
            feature_cols = [feature_cols[i] for i in classifier.selected_features]
        
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # Save model
    print("\nSaving model...")
    classifier.save()
    
    # Save train/val/test splits
    print("\nSaving train/val/test splits...")
    np.save(Config.X_TRAIN_NPY, X_train)
    np.save(Config.X_VAL_NPY, X_val)
    np.save(Config.X_TEST_NPY, X_test)
    np.save(Config.Y_TRAIN_NPY, y_train)
    np.save(Config.Y_VAL_NPY, y_val)
    np.save(Config.Y_TEST_NPY, y_test)
    
    print(f"Saved splits to {Config.PROCESSED_DATA_DIR}")
    
    print("\n" + "=" * 60)
    print("Step 5 Complete!")
    print("=" * 60)
    print(f"\nOptimal threshold: {classifier.optimal_threshold:.2f}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    
    # Porównanie z domyślnym progiem 0.5
    if classifier.optimal_threshold != 0.5:
        print("\n[Comparison with default threshold 0.5]")
        y_pred_default = (classifier.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        f1_default = f1_score(y_test, y_pred_default, zero_division=0)
        print(f"Test F1 (threshold=0.5): {f1_default:.4f}")
        print(f"Test F1 (threshold={classifier.optimal_threshold:.2f}): {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
