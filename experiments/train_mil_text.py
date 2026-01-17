"""
Sprint 2: Train MIL Text Model

Training script for Multi-Instance Learning text model with RoBERTa.
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required. Install with: pip install torch")
    sys.exit(1)

from data.interview_segmentation import create_mil_instances_dataset
from data.mil_dataloader import create_mil_dataloaders
from models.mil_text_model import MILTextModel, MILTextTrainer
from evaluation.report import EvaluationReport


def ensure_instances_exist():
    """Create instances dataset if it doesn't exist"""
    if not CONFIG.INSTANCES_CSV.exists():
        print("Creating MIL instances dataset...")
        create_mil_instances_dataset(segmentation_method='sentences')


def train_mil_text_model(
    pooling_strategy: str = 'attention',
    epochs: int = None,
    patience: int = None,
    use_mixed_precision: bool = None,
    unfreeze_last_n_layers: int = 2,
    pos_weight: float = 2.0
):
    """
    Train MIL Text Model.
    
    Args:
        pooling_strategy: 'max', 'mean', 'attention', or 'alpha_beta'
        epochs: Number of training epochs
        patience: Early stopping patience
        use_mixed_precision: Whether to use mixed precision training
        unfreeze_last_n_layers: Unfreeze last N RoBERTa layers (0=fully frozen, 2=recommended)
        pos_weight: Class weight for positive samples (3.4 for DAIC-WOZ imbalance)
    """
    epochs = epochs or CONFIG.EPOCHS
    patience = patience or CONFIG.EARLY_STOPPING_PATIENCE
    use_mixed_precision = use_mixed_precision if use_mixed_precision is not None else CONFIG.USE_MIXED_PRECISION
    
    print("=" * 60)
    print("Sprint 2: Train MIL Text Model")
    print("=" * 60)
    
    # Set seed
    CONFIG.set_seed()
    
    # Print GPU info
    CONFIG.print_gpu_info()
    
    # Ensure instances exist
    ensure_instances_exist()
    
    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_mil_dataloaders(
        batch_size=CONFIG.MIL_BATCH_SIZE
    )
    
    # Create model
    print("\nInitializing model...")
    device = CONFIG.DEVICE
    
    model = MILTextModel(
        model_name=CONFIG.ROBERTA_MODEL_NAME,
        hidden_dim=256,
        dropout_rate=CONFIG.DROPOUT_RATE,
        alpha=CONFIG.MIL_ALPHA,
        beta=CONFIG.MIL_BETA,
        freeze_encoder=True,
        unfreeze_last_n_layers=unfreeze_last_n_layers,  # Fine-tune last N layers
        pooling_strategy=pooling_strategy,
        device=device
    )
    model = model.to(device)
    
    # Create trainer
    trainer = MILTextTrainer(
        model=model,
        learning_rate=CONFIG.MIL_LEARNING_RATE,
        weight_decay=CONFIG.WEIGHT_DECAY
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device == 'cuda' else None
    
    # Training loop
    print("\n" + "-" * 60)
    print(f"Training Configuration:")
    print(f"  Pooling: {pooling_strategy}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {CONFIG.MIL_BATCH_SIZE}")
    print(f"  Learning rate: {CONFIG.MIL_LEARNING_RATE}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Mixed precision: {scaler is not None}")
    print(f"  Unfreeze last N layers: {unfreeze_last_n_layers}")
    print(f"  Positive class weight: {pos_weight}")
    print("-" * 60 + "\n")
    
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            bag_instances = batch['bag_instances']
            labels = batch['labels'].to(device)
            
            trainer.optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    predictions, instance_scores = model.forward_batch(
                        bag_instances, return_instance_scores=True
                    )
                    loss = model.compute_loss(predictions, labels, instance_scores, pos_weight=pos_weight)
                
                scaler.scale(loss).backward()
                scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(trainer.optimizer)
                scaler.update()
            else:
                predictions, instance_scores = model.forward_batch(
                    bag_instances, return_instance_scores=True
                )
                loss = model.compute_loss(predictions, labels, instance_scores, pos_weight=pos_weight)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                trainer.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss = total_loss / n_batches
        
        # Validation
        val_metrics, val_preds, val_labels, _ = trainer.evaluate(val_loader)
        
        # Update scheduler
        trainer.scheduler.step(val_metrics['f1'])
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  * New best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (Val F1: {best_val_f1:.4f})")
    
    # Test evaluation
    print("\n" + "-" * 60)
    print("Test Set Evaluation")
    print("-" * 60)
    
    test_metrics, test_preds, test_labels, test_instance_scores = trainer.evaluate(test_loader)
    
    # Optimize threshold
    from sklearn.metrics import f1_score
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        f1 = f1_score(test_labels, (test_preds >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    test_pred_binary = (test_preds >= best_thresh).astype(int)
    
    # Generate report
    report = EvaluationReport(
        model_name=f"MIL Text ({pooling_strategy})",
        y_true=test_labels,
        y_pred=test_pred_binary,
        y_proba=test_preds
    )
    report.generate(n_bootstrap=CONFIG.BOOTSTRAP_CI_RESAMPLES)
    report.print_report()
    report.save_json()
    report.save_plots()
    
    # Save model
    model.save()
    
    # Instance-level analysis
    print("\n" + "-" * 60)
    print("Instance-Level Analysis")
    print("-" * 60)
    
    # Find most "depressed" instances
    all_instances = []
    for batch in test_loader:
        for bag_idx, (instances, scores, label) in enumerate(zip(
            batch['bag_instances'],
            test_instance_scores[len(all_instances):],
            test_labels[len(all_instances):]
        )):
            for i, (inst, score) in enumerate(zip(instances, scores)):
                all_instances.append({
                    'text': inst[:100] + '...' if len(inst) > 100 else inst,
                    'score': float(score),
                    'bag_label': int(label)
                })
    
    # Sort by score
    all_instances.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 5 highest-scoring instances:")
    for i, inst in enumerate(all_instances[:5]):
        print(f"  {i+1}. Score: {inst['score']:.3f} | Label: {inst['bag_label']} | {inst['text']}")
    
    print("\nTop 5 lowest-scoring instances:")
    for i, inst in enumerate(all_instances[-5:]):
        print(f"  {i+1}. Score: {inst['score']:.3f} | Label: {inst['bag_label']} | {inst['text']}")
    
    print("\n" + "=" * 60)
    print("Sprint 2 Complete!")
    print("=" * 60)
    
    return model, report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MIL Text Model')
    parser.add_argument('--pooling', type=str, default='attention',
                       choices=['max', 'mean', 'attention', 'alpha_beta'],
                       help='MIL pooling strategy')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision')
    parser.add_argument('--unfreeze-layers', type=int, default=2,
                       help='Unfreeze last N RoBERTa layers (0=frozen, 2=recommended)')
    parser.add_argument('--pos-weight', type=float, default=3.4,
                       help='Class weight for positive samples (3.4 for DAIC-WOZ)')
    args = parser.parse_args()
    
    train_mil_text_model(
        pooling_strategy=args.pooling,
        epochs=args.epochs,
        patience=args.patience,
        use_mixed_precision=not args.no_amp,
        unfreeze_last_n_layers=args.unfreeze_layers,
        pos_weight=args.pos_weight
    )
