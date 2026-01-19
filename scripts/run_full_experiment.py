#!/usr/bin/env python
"""
MDDM Full Experiment Pipeline
=============================

Kompletny skrypt przeprowadzający cały eksperyment od A do Z:
1. Walidacja danych
2. Ekstrakcja features (BERT text + Wav2Vec audio + prosody)
3. Trening modeli (XGBoost Early Fusion, Late Fusion)
4. Kalibracja (Platt Scaling)
5. Ablation Study
6. Generowanie raportu

Użycie:
    python scripts/run_full_experiment.py
    python scripts/run_full_experiment.py --skip-audio  # Pomiń Wav2Vec (szybciej)
    python scripts/run_full_experiment.py --quick       # Szybki test (3-fold CV)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import gc
warnings.filterwarnings('ignore')

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import CONFIG

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> str:
    """Get GPU memory usage info."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "GPU not available"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / 'logs' / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Main experiment pipeline class."""
    
    def __init__(
        self,
        data_dir: Path = None,
        output_dir: Path = None,
        use_wav2vec: bool = True,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.data_dir = data_dir or ROOT_DIR / 'data' / 'raw'
        self.splits_dir = ROOT_DIR / 'DAIC-WOZ-Dataset'
        self.output_dir = output_dir or ROOT_DIR / 'results'
        self.processed_dir = ROOT_DIR / 'data' / 'processed'
        
        self.use_wav2vec = use_wav2vec
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (ROOT_DIR / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.features = {}
        
        # Setup GPU memory management
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU with memory limits for RTX 5060 8GB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("=" * 50)
            logger.info("GPU SETUP")
            logger.info("=" * 50)
            
            # Print GPU info
            CONFIG.print_gpu_info()
            
            # Set memory limit
            try:
                CONFIG.setup_gpu_memory_limit()
            except Exception as e:
                logger.warning(f"Could not setup GPU memory limit: {e}")
            
            logger.info(get_gpu_memory_info())
            logger.info("=" * 50)
        else:
            logger.info("Running on CPU (no GPU available)")
    
    def run(self):
        """Run the full pipeline."""
        logger.info("=" * 70)
        logger.info("MDDM FULL EXPERIMENT PIPELINE")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        try:
            # Step 1: Validate data
            self.step_1_validate_data()
            
            # Step 2: Load labels
            self.step_2_load_labels()
            
            # Step 3: Extract features
            self.step_3_extract_features()
            
            # Step 4: Train models
            self.step_4_train_models()
            
            # Step 5: Ablation study
            self.step_5_ablation_study()
            
            # Step 6: Generate report
            self.step_6_generate_report()
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def step_1_validate_data(self):
        """Validate that all required data files exist."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Validating Data")
        logger.info("=" * 50)
        
        # Check split files
        train_split = self.splits_dir / 'train_split_Depression_AVEC2017.csv'
        dev_split = self.splits_dir / 'dev_split_Depression_AVEC2017.csv'
        test_split = self.splits_dir / 'test_split_Depression_AVEC2017.csv'
        
        for split_file in [train_split, dev_split, test_split]:
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load participant IDs from splits
        train_df = pd.read_csv(train_split)
        dev_df = pd.read_csv(dev_split)
        test_df = pd.read_csv(test_split)
        
        all_participants = set(train_df['Participant_ID'].tolist() + 
                              dev_df['Participant_ID'].tolist() +
                              test_df['Participant_ID'].tolist())
        
        logger.info(f"Total participants in splits: {len(all_participants)}")
        logger.info(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        
        # Check audio and transcript files
        missing_audio = []
        missing_transcript = []
        available_participants = []
        
        for pid in all_participants:
            audio_file = self.data_dir / f"{pid}_AUDIO.wav"
            transcript_file = self.data_dir / f"{pid}_TRANSCRIPT.csv"
            
            has_audio = audio_file.exists()
            has_transcript = transcript_file.exists()
            
            if not has_audio:
                missing_audio.append(pid)
            if not has_transcript:
                missing_transcript.append(pid)
            
            if has_audio or has_transcript:
                available_participants.append(pid)
        
        logger.info(f"\nData availability:")
        logger.info(f"  Audio files: {len(all_participants) - len(missing_audio)}/{len(all_participants)}")
        logger.info(f"  Transcript files: {len(all_participants) - len(missing_transcript)}/{len(all_participants)}")
        
        if missing_audio:
            logger.warning(f"  Missing audio for: {missing_audio[:5]}...")
        if missing_transcript:
            logger.warning(f"  Missing transcripts for: {missing_transcript[:5]}...")
        
        self.all_participants = sorted(available_participants)
        self.train_ids = [p for p in train_df['Participant_ID'] if p in self.all_participants]
        self.dev_ids = [p for p in dev_df['Participant_ID'] if p in self.all_participants]
        self.test_ids = [p for p in test_df['Participant_ID'] if p in self.all_participants]
        
        logger.info(f"\nUsable participants: {len(self.all_participants)}")
        logger.info("[OK] Data validation complete!")
    
    def step_2_load_labels(self):
        """Load labels from split files."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Loading Labels")
        logger.info("=" * 50)
        
        train_df = pd.read_csv(self.splits_dir / 'train_split_Depression_AVEC2017.csv')
        dev_df = pd.read_csv(self.splits_dir / 'dev_split_Depression_AVEC2017.csv')
        test_df = pd.read_csv(self.splits_dir / 'test_split_Depression_AVEC2017.csv')
        
        # Combine all labels
        all_labels_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
        
        # Create label dict
        self.labels = {}
        for _, row in all_labels_df.iterrows():
            pid = row['Participant_ID']
            if pid in self.all_participants:
                self.labels[pid] = {
                    'binary': int(row['PHQ8_Binary']),
                    'score': int(row['PHQ8_Score']) if pd.notna(row['PHQ8_Score']) else 0,
                    'gender': int(row['Gender']) if pd.notna(row['Gender']) else 0
                }
        
        # Stats
        n_positive = sum(1 for p in self.labels.values() if p['binary'] == 1)
        n_negative = len(self.labels) - n_positive
        
        logger.info(f"Labels loaded: {len(self.labels)} participants")
        logger.info(f"  Positive (depressed): {n_positive} ({n_positive/len(self.labels)*100:.1f}%)")
        logger.info(f"  Negative (healthy): {n_negative} ({n_negative/len(self.labels)*100:.1f}%)")
        
        self.pos_weight = n_negative / n_positive
        logger.info(f"  Class weight (pos_weight): {self.pos_weight:.2f}")
        
        # Save labels
        labels_df = pd.DataFrame([
            {'participant_id': pid, **data} 
            for pid, data in self.labels.items()
        ])
        labels_df.to_csv(self.processed_dir / 'labels.csv', index=False)
        
        logger.info("[OK] Labels loaded!")
    
    def step_3_extract_features(self):
        """Extract text and audio features."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Extracting Features")
        logger.info("=" * 50)
        
        # 3a: Text features
        self._extract_text_features()
        
        # 3b: Audio features
        self._extract_audio_features()
        
        logger.info("[OK] Feature extraction complete!")
    
    def _extract_text_features(self):
        """Extract advanced text features (BERT + topic + linguistic)."""
        logger.info("\n--- Extracting Text Features (Advanced: BERT + Topics + Linguistic) ---")
        logger.info(get_gpu_memory_info())
        
        from models.text_encoder_bert import TextEncoderAdvanced
        
        # Use advanced encoder with all features
        encoder = TextEncoderAdvanced(
            bert_model='multilingual',
            n_topics=30,  # Reduced for small dataset
            use_topics=True,
            use_linguistic=True,
            use_sentiment=True
        )
        logger.info(f"Using: {encoder}")
        
        # First pass: collect all transcripts for topic model fitting
        all_transcripts = []
        transcript_map = {}
        participant_ids = []
        
        for pid in self.all_participants:
            transcript_file = self.data_dir / f"{pid}_TRANSCRIPT.csv"
            participant_ids.append(pid)
            
            if not transcript_file.exists():
                transcript_map[pid] = ""
                continue
            
            try:
                df = pd.read_csv(transcript_file, sep='\t')
                
                if 'speaker' in df.columns:
                    participant_text = df[df['speaker'] == 'Participant']['value'].tolist()
                elif 'value' in df.columns:
                    participant_text = df['value'].tolist()
                else:
                    participant_text = df.iloc[:, -1].tolist()
                
                full_text = ' '.join([str(t) for t in participant_text if pd.notna(t)])
                transcript_map[pid] = full_text
                
                if full_text.strip():
                    all_transcripts.append(full_text)
                    
            except Exception as e:
                logger.warning(f"Error loading transcript {pid}: {e}")
                transcript_map[pid] = ""
        
        # Fit topic model on all transcripts
        if all_transcripts:
            logger.info(f"Fitting topic model on {len(all_transcripts)} transcripts...")
            encoder.fit_topics(all_transcripts)
        
        # Second pass: extract features
        text_features = []
        
        for i, pid in enumerate(self.all_participants):
            full_text = transcript_map.get(pid, "")
            
            try:
                if not full_text.strip():
                    text_features.append(np.zeros(encoder.get_embedding_dim()))
                else:
                    embedding = encoder.encode_all([full_text])[0]
                    text_features.append(embedding)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM for {pid}, clearing memory")
                    clear_gpu_memory()
                    text_features.append(np.zeros(encoder.get_embedding_dim()))
                else:
                    logger.warning(f"Error processing {pid}: {e}")
                    text_features.append(np.zeros(encoder.get_embedding_dim()))
            except Exception as e:
                logger.warning(f"Error processing {pid}: {e}")
                text_features.append(np.zeros(encoder.get_embedding_dim()))
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i+1}/{len(self.all_participants)} transcripts")
                clear_gpu_memory()
        
        self.features['text'] = np.array(text_features)
        self.features['participant_ids'] = participant_ids
        
        clear_gpu_memory()
        
        np.save(self.processed_dir / 'text_features.npy', self.features['text'])
        logger.info(f"Text features shape: {self.features['text'].shape}")
        logger.info(get_gpu_memory_info())
    
    def _extract_audio_features(self):
        """Extract advanced audio features (VAD + Wav2Vec + MFCC + Glottal + Prosody)."""
        logger.info(f"\n--- Extracting Audio Features (Advanced: VAD + Wav2Vec={self.use_wav2vec}) ---")
        logger.info(get_gpu_memory_info())
        
        from models.audio_encoder_advanced import AudioEncoderAdvanced
        
        # Use advanced encoder with VAD and all features
        encoder = AudioEncoderAdvanced(
            use_wav2vec=self.use_wav2vec,
            use_vad=True,
            use_extended_mfcc=True,
            use_glottal=True,
            use_prosody=True,
            vad_backend='energy'  # Use energy-based VAD (faster, no extra deps)
        )
        logger.info(f"Using: {encoder}")
        
        audio_features = []
        failed_count = 0
        
        for i, pid in enumerate(self.all_participants):
            audio_file = self.data_dir / f"{pid}_AUDIO.wav"
            
            if not audio_file.exists():
                logger.warning(f"Missing audio for {pid}, using zeros")
                audio_features.append(np.zeros(encoder.get_embedding_dim()))
                continue
            
            try:
                embedding = encoder.extract_all(str(audio_file))
                audio_features.append(embedding)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM for audio {pid}, clearing memory and retrying")
                    clear_gpu_memory()
                    
                    try:
                        embedding = encoder.extract_all(str(audio_file))
                        audio_features.append(embedding)
                    except Exception as e2:
                        logger.error(f"Retry failed for {pid}: {e2}")
                        audio_features.append(np.zeros(encoder.get_embedding_dim()))
                        failed_count += 1
                else:
                    logger.warning(f"Error processing audio {pid}: {e}")
                    audio_features.append(np.zeros(encoder.get_embedding_dim()))
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing audio {pid}: {e}")
                audio_features.append(np.zeros(encoder.get_embedding_dim()))
                failed_count += 1
            
            if (i + 1) % 5 == 0:
                clear_gpu_memory()
                
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(self.all_participants)} audio files")
                logger.debug(get_gpu_memory_info())
        
        self.features['audio'] = np.array(audio_features)
        
        clear_gpu_memory()
        
        np.save(self.processed_dir / 'audio_features.npy', self.features['audio'])
        logger.info(f"Audio features shape: {self.features['audio'].shape}")
        logger.info(f"Failed extractions: {failed_count}/{len(self.all_participants)}")
        logger.info(get_gpu_memory_info())
    
    def step_4_train_models(self):
        """Train and evaluate models including attention-based fusion."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: Training Models")
        logger.info("=" * 50)
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X_text = self.features['text']
        X_audio = self.features['audio']
        X_combined = np.hstack([X_audio, X_text])
        
        y = np.array([self.labels[pid]['binary'] for pid in self.features['participant_ids']])
        
        logger.info(f"Data shapes:")
        logger.info(f"  X_audio: {X_audio.shape}")
        logger.info(f"  X_text: {X_text.shape}")
        logger.info(f"  X_combined: {X_combined.shape}")
        logger.info(f"  y: {y.shape} ({y.mean():.1%} positive)")
        
        # Store dimensions for neural network models
        self.audio_dim = X_audio.shape[1]
        self.text_dim = X_text.shape[1]
        
        # Train XGBoost (Early Fusion)
        self._train_xgboost(X_combined, y, "xgboost_early_fusion")
        
        # Train with text only
        self._train_xgboost(X_text, y, "xgboost_text_only")
        
        # Train with audio only
        self._train_xgboost(X_audio, y, "xgboost_audio_only")
        
        # Train Attention Fusion Model (new SOTA approach)
        self._train_attention_fusion(X_audio, X_text, y, "attention_fusion")
        
        # Train Normalized Late Fusion Model
        self._train_normalized_fusion(X_audio, X_text, y, "normalized_fusion")
        
        logger.info("[OK] Model training complete!")
    
    def _train_attention_fusion(
        self,
        X_audio: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        model_name: str
    ):
        """Train Attention-based Fusion Model with cross-validation."""
        logger.info(f"\n--- Training {model_name} (Attention-based) ---")
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping attention fusion")
            return
        
        from models.attention_fusion import AttentionFusionModel, AttentionFusionTrainer
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, DataLoader
        
        # Scale features
        scaler_audio = StandardScaler()
        scaler_text = StandardScaler()
        X_audio_scaled = scaler_audio.fit_transform(X_audio)
        X_text_scaled = scaler_text.fit_transform(X_text)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'f1': [], 'auroc': [], 'mcc': [],
            'specificity': [], 'sensitivity': [],
            'audio_weight': [], 'text_weight': []
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_audio_scaled, y)):
            # Prepare data
            X_audio_train = torch.FloatTensor(X_audio_scaled[train_idx])
            X_audio_val = torch.FloatTensor(X_audio_scaled[val_idx])
            X_text_train = torch.FloatTensor(X_text_scaled[train_idx])
            X_text_val = torch.FloatTensor(X_text_scaled[val_idx])
            y_train = torch.FloatTensor(y[train_idx])
            y_val = torch.FloatTensor(y[val_idx])
            
            train_dataset = TensorDataset(X_audio_train, X_text_train, y_train)
            val_dataset = TensorDataset(X_audio_val, X_text_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Create model
            model = AttentionFusionModel(
                audio_dim=self.audio_dim,
                text_dim=self.text_dim,
                hidden_dim=128,  # Smaller for limited data
                num_heads=4,
                num_attention_layers=1,
                dropout=0.4,
                use_cross_attention=True,
                use_gated_fusion=True
            )
            
            # Create trainer
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            
            trainer = AttentionFusionTrainer(
                model=model,
                n_positive=max(1, n_pos),
                n_negative=max(1, n_neg),
                learning_rate=1e-4,
                weight_decay=0.01,
                device=device
            )
            
            # Train
            history = trainer.train(
                train_loader,
                val_loader,
                epochs=50,
                patience=10,
                verbose=False
            )
            
            # Evaluate
            metrics = trainer.evaluate(val_loader)
            
            fold_results['f1'].append(metrics['f1'])
            fold_results['auroc'].append(metrics.get('auroc', 0.5))
            fold_results['mcc'].append(metrics['mcc'])
            fold_results['specificity'].append(metrics['specificity'])
            fold_results['sensitivity'].append(metrics['recall'])
            fold_results['audio_weight'].append(metrics.get('audio_weight', 0.5))
            fold_results['text_weight'].append(metrics.get('text_weight', 0.5))
            
            logger.info(f"  Fold {fold+1}: F1={metrics['f1']:.3f}, "
                       f"MCC={metrics['mcc']:.3f}, "
                       f"Audio={metrics.get('audio_weight', 0.5):.2f}")
            
            # Cleanup
            del model, trainer
            clear_gpu_memory()
        
        # Aggregate results
        self.results[model_name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
            for metric, values in fold_results.items()
        }
        
        logger.info(f"\n  {model_name} SUMMARY:")
        logger.info(f"    F1: {np.mean(fold_results['f1']):.3f} +/- {np.std(fold_results['f1']):.3f}")
        logger.info(f"    MCC: {np.mean(fold_results['mcc']):.3f} +/- {np.std(fold_results['mcc']):.3f}")
        logger.info(f"    Avg Audio Weight: {np.mean(fold_results['audio_weight']):.3f}")
        logger.info(f"    Avg Text Weight: {np.mean(fold_results['text_weight']):.3f}")
    
    def _train_normalized_fusion(
        self,
        X_audio: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        model_name: str
    ):
        """Train Normalized Late Fusion Model with cross-validation."""
        logger.info(f"\n--- Training {model_name} (Normalized with Gating) ---")
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping normalized fusion")
            return
        
        from models.fusion_fixed import NormalizedLateFusionModel, FusionTrainer
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, DataLoader
        
        # Scale features
        scaler_audio = StandardScaler()
        scaler_text = StandardScaler()
        X_audio_scaled = scaler_audio.fit_transform(X_audio)
        X_text_scaled = scaler_text.fit_transform(X_text)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'f1': [], 'auroc': [], 'mcc': [],
            'specificity': [], 'sensitivity': []
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_audio_scaled, y)):
            # Prepare data
            X_audio_train = torch.FloatTensor(X_audio_scaled[train_idx])
            X_audio_val = torch.FloatTensor(X_audio_scaled[val_idx])
            X_text_train = torch.FloatTensor(X_text_scaled[train_idx])
            X_text_val = torch.FloatTensor(X_text_scaled[val_idx])
            y_train = torch.FloatTensor(y[train_idx])
            y_val = torch.FloatTensor(y[val_idx])
            
            train_dataset = TensorDataset(X_audio_train, X_text_train, y_train)
            val_dataset = TensorDataset(X_audio_val, X_text_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Create model
            model = NormalizedLateFusionModel(
                audio_dim=self.audio_dim,
                text_dim=self.text_dim,
                hidden_dim=128,
                use_gating=True,
                dropout=0.4
            )
            
            # Create trainer
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            
            trainer = FusionTrainer(
                model=model,
                n_positive=max(1, n_pos),
                n_negative=max(1, n_neg),
                learning_rate=1e-3,
                weight_decay=0.01,
                device=device
            )
            
            # Train
            history = trainer.train(
                train_loader,
                val_loader,
                epochs=50,
                patience=10,
                verbose=False
            )
            
            # Evaluate
            metrics = trainer.evaluate(val_loader)
            
            fold_results['f1'].append(metrics['f1'])
            fold_results['mcc'].append(metrics['mcc'])
            fold_results['specificity'].append(metrics['specificity'])
            fold_results['sensitivity'].append(metrics['recall'])
            
            # Compute AUROC
            model.eval()
            with torch.no_grad():
                y_proba = model.predict_proba(X_audio_val.to(device), X_text_val.to(device))
                y_proba_np = y_proba.cpu().numpy()
            
            from sklearn.metrics import roc_auc_score
            try:
                auroc = roc_auc_score(y[val_idx], y_proba_np)
            except:
                auroc = 0.5
            fold_results['auroc'].append(auroc)
            
            logger.info(f"  Fold {fold+1}: F1={metrics['f1']:.3f}, "
                       f"AUROC={auroc:.3f}, "
                       f"MCC={metrics['mcc']:.3f}")
            
            # Cleanup
            del model, trainer
            clear_gpu_memory()
        
        # Aggregate results
        self.results[model_name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
            for metric, values in fold_results.items()
        }
        
        logger.info(f"\n  {model_name} SUMMARY:")
        logger.info(f"    F1: {np.mean(fold_results['f1']):.3f} +/- {np.std(fold_results['f1']):.3f}")
        logger.info(f"    AUROC: {np.mean(fold_results['auroc']):.3f} +/- {np.std(fold_results['auroc']):.3f}")
        logger.info(f"    MCC: {np.mean(fold_results['mcc']):.3f} +/- {np.std(fold_results['mcc']):.3f}")
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, model_name: str):
        """Train XGBoost with cross-validation and calibration."""
        logger.info(f"\n--- Training {model_name} ---")
        
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not installed, using RandomForest")
            from sklearn.ensemble import RandomForestClassifier as XGBClassifier
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
        from evaluation.calibration import calibrate_model, expected_calibration_error
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'f1': [], 'auroc': [], 'mcc': [], 
            'specificity': [], 'sensitivity': [],
            'ece_before': [], 'ece_after': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=self.pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Calibration
            try:
                calibrated = calibrate_model(model, X_val, y_val)
                y_proba_cal = calibrated.predict_proba(X_val)[:, 1]
                ece_before = expected_calibration_error(y_val, y_proba)
                ece_after = expected_calibration_error(y_val, y_proba_cal)
            except:
                ece_before = ece_after = 0.1
            
            # Metrics
            tn = ((y_val == 0) & (y_pred == 0)).sum()
            fp = ((y_val == 0) & (y_pred == 1)).sum()
            tp = ((y_val == 1) & (y_pred == 1)).sum()
            fn = ((y_val == 1) & (y_pred == 0)).sum()
            
            fold_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            fold_results['auroc'].append(roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5)
            fold_results['mcc'].append(matthews_corrcoef(y_val, y_pred))
            fold_results['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            fold_results['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_results['ece_before'].append(ece_before)
            fold_results['ece_after'].append(ece_after)
            
            logger.info(f"  Fold {fold+1}: F1={fold_results['f1'][-1]:.3f}, "
                       f"AUROC={fold_results['auroc'][-1]:.3f}, "
                       f"MCC={fold_results['mcc'][-1]:.3f}")
        
        # Aggregate results
        self.results[model_name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
            for metric, values in fold_results.items()
        }
        
        logger.info(f"\n  {model_name} SUMMARY:")
        logger.info(f"    F1: {np.mean(fold_results['f1']):.3f} +/- {np.std(fold_results['f1']):.3f}")
        logger.info(f"    AUROC: {np.mean(fold_results['auroc']):.3f} +/- {np.std(fold_results['auroc']):.3f}")
        logger.info(f"    MCC: {np.mean(fold_results['mcc']):.3f} +/- {np.std(fold_results['mcc']):.3f}")
        logger.info(f"    Specificity: {np.mean(fold_results['specificity']):.3f}")
        logger.info(f"    ECE: {np.mean(fold_results['ece_before']):.3f} -> {np.mean(fold_results['ece_after']):.3f}")
    
    def step_5_ablation_study(self):
        """Run ablation study."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: Ablation Study")
        logger.info("=" * 50)
        
        from evaluation.ablation import AblationStudy
        
        X_text = self.features['text']
        X_audio = self.features['audio']
        y = np.array([self.labels[pid]['binary'] for pid in self.features['participant_ids']])
        
        try:
            from xgboost import XGBClassifier
            def model_factory():
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    scale_pos_weight=self.pos_weight,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1
                )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            def model_factory():
                return RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=self.random_state
                )
        
        ablation = AblationStudy(X_audio, X_text, y)
        results_df = ablation.run_full_ablation(model_factory, n_splits=self.n_splits)
        
        # Save results
        results_df.to_csv(self.output_dir / 'ablation_results.csv')
        ablation.save_results(self.output_dir / 'ablation')
        
        # Print report
        print(ablation.generate_report())
        
        self.ablation_results = ablation.results
        
        logger.info("[OK] Ablation study complete!")
    
    def step_6_generate_report(self):
        """Generate final experiment report."""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 6: Generating Report")
        logger.info("=" * 50)
        
        # Save all results
        with open(self.output_dir / 'experiment_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'n_participants': len(self.all_participants),
                    'n_splits': self.n_splits,
                    'use_wav2vec': self.use_wav2vec,
                    'pos_weight': self.pos_weight
                },
                'model_results': self.results,
            }, f, indent=2)
        
        # Generate text report
        report = self._generate_text_report()
        
        with open(self.output_dir / 'experiment_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info("[OK] Report generated!")
    
    def _generate_text_report(self) -> str:
        """Generate text summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("MDDM EXPERIMENT REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        lines.append(f"\nDATASET:")
        lines.append(f"  Participants: {len(self.all_participants)}")
        lines.append(f"  Positive (depressed): {sum(1 for p in self.labels.values() if p['binary']==1)}")
        lines.append(f"  Negative (healthy): {sum(1 for p in self.labels.values() if p['binary']==0)}")
        
        lines.append(f"\nFEATURES:")
        lines.append(f"  Audio: {self.features['audio'].shape[1]} dimensions")
        lines.append(f"  Text: {self.features['text'].shape[1]} dimensions")
        lines.append(f"  Wav2Vec used: {self.use_wav2vec}")
        
        lines.append(f"\nMODEL RESULTS ({self.n_splits}-fold CV):")
        lines.append("-" * 70)
        lines.append(f"{'Model':<30} {'F1':>8} {'AUROC':>8} {'MCC':>8} {'Spec':>8}")
        lines.append("-" * 70)
        
        for model_name, metrics in self.results.items():
            f1 = metrics['f1']['mean']
            auroc = metrics['auroc']['mean']
            mcc = metrics['mcc']['mean']
            spec = metrics['specificity']['mean']
            lines.append(f"{model_name:<30} {f1:>8.3f} {auroc:>8.3f} {mcc:>8.3f} {spec:>8.3f}")
        
        lines.append("-" * 70)
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1']['mean'])
        lines.append(f"\nBEST MODEL: {best_model[0]}")
        lines.append(f"  F1 = {best_model[1]['f1']['mean']:.3f} +/- {best_model[1]['f1']['std']:.3f}")
        
        # Success criteria check
        lines.append(f"\nSUCCESS CRITERIA CHECK:")
        f1_best = best_model[1]['f1']['mean']
        auroc_best = best_model[1]['auroc']['mean']
        mcc_best = best_model[1]['mcc']['mean']
        spec_best = best_model[1]['specificity']['mean']
        
        lines.append(f"  F1 >= 0.65: {'PASS' if f1_best >= 0.65 else 'FAIL'} ({f1_best:.3f})")
        lines.append(f"  AUROC >= 0.75: {'PASS' if auroc_best >= 0.75 else 'FAIL'} ({auroc_best:.3f})")
        lines.append(f"  MCC > 0: {'PASS' if mcc_best > 0 else 'FAIL'} ({mcc_best:.3f})")
        lines.append(f"  Specificity >= 0.70: {'PASS' if spec_best >= 0.70 else 'FAIL'} ({spec_best:.3f})")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run full MDDM experiment')
    parser.add_argument('--skip-audio', action='store_true',
                       help='Skip Wav2Vec (use only prosody features)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 3-fold CV')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    pipeline = ExperimentPipeline(
        use_wav2vec=not args.skip_audio,
        n_splits=3 if args.quick else 5,
        output_dir=args.output,
        random_state=args.seed
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
