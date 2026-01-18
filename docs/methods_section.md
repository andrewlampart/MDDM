# Methods

## 2.1 Dataset

We utilized the DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz) dataset, a publicly available resource for depression detection research. The dataset consists of semi-structured clinical interviews conducted by an animated virtual interviewer, designed to support the diagnosis of psychological distress conditions including depression, anxiety, and post-traumatic stress disorder.

### 2.1.1 Data Composition

The DAIC-WOZ corpus contains 189 participants (107 train, 35 development, 47 test) with associated audio recordings, transcripts, and facial feature extractions. Each participant completed the Patient Health Questionnaire-8 (PHQ-8), providing a standardized measure of depression severity. Following previous studies, we binarized PHQ-8 scores using a threshold of 10, where scores ≥10 indicate moderate or greater depression severity (positive class) and scores <10 indicate minimal to mild symptoms (negative class).

The dataset exhibits class imbalance with approximately 30% positive cases (depressed) and 70% negative cases (non-depressed). We address this imbalance through class-weighted loss functions and stratified sampling during cross-validation.

### 2.1.2 Data Splits

We followed the standard DAIC-WOZ train/development/test split provided by the AVEC 2017 challenge organizers. For model development and hyperparameter tuning, we employed 5-fold stratified cross-validation on the combined train and development sets, reserving the test set for final evaluation. This approach ensures robust performance estimates while maintaining comparability with prior work.

## 2.2 Feature Extraction

### 2.2.1 Text Features

Depression often manifests in language through changes in content, sentiment, and linguistic patterns. To capture these markers, we employed pretrained transformer-based language models that have demonstrated strong performance across diverse NLP tasks.

**Text Encoder**: We used SentenceTransformer with the `distiluse-base-multilingual-cased-v2` model, a distilled version of Universal Sentence Encoder fine-tuned for semantic similarity tasks. This model produces 512-dimensional dense embeddings that capture semantic meaning at the sentence level.

The choice of a pretrained encoder offers several advantages over training embeddings from scratch:
1. **Transfer learning**: The model has been pretrained on over 1 billion sentence pairs, providing robust linguistic representations
2. **Reduced overfitting**: Fixed embeddings eliminate trainable text parameters, critical given our limited sample size (n=189)
3. **Multilingual support**: The model handles multiple languages, enabling future extensions to non-English datasets

For each participant, we encoded the full interview transcript and computed a single embedding vector by mean pooling over sentence embeddings when transcripts exceeded maximum token limits.

### 2.2.2 Audio Features

Vocal characteristics provide complementary information to linguistic content, as depression is associated with measurable changes in speech prosody, voice quality, and temporal dynamics.

**Wav2Vec 2.0 Embeddings**: We employed the `facebook/wav2vec2-base-960h` model, a self-supervised speech representation model pretrained on 960 hours of LibriSpeech audio. The model produces 768-dimensional embeddings capturing acoustic patterns without requiring task-specific training. We applied mean pooling over the temporal dimension to obtain fixed-size representations.

**Prosody Features**: To ensure interpretability and capture depression-specific acoustic markers, we extracted a complementary set of hand-crafted prosody features:

- **Fundamental Frequency (F0)**: Mean, standard deviation, and range of F0, extracted using the probabilistic YIN algorithm. Reduced F0 variation ("flat affect") is a known depression marker.
- **Energy Contour**: Mean, standard deviation, and range of root-mean-square energy, capturing speech intensity patterns.
- **Speech Rate**: Zero-crossing rate statistics as a proxy for articulation rate. Slower speech and increased pausing are associated with depression.
- **Spectral Features**: Spectral centroid, rolloff, and bandwidth capturing voice quality characteristics.
- **MFCC Dynamics**: Mean temporal variation of Mel-frequency cepstral coefficients, reflecting voice quality stability.

The combined audio representation consists of 768 Wav2Vec dimensions plus approximately 18 prosody features, yielding ~786 total audio features.

## 2.3 Multimodal Fusion

We investigated multiple fusion strategies to combine audio and text modalities:

### 2.3.1 Early Fusion (Feature Concatenation)

In early fusion, audio and text feature vectors are concatenated to form a single multimodal representation, which is then processed by a gradient boosting classifier (XGBoost). This approach assumes that meaningful interactions between modalities can be learned by the classifier without explicit modeling.

The concatenated feature vector has dimensionality $d_{audio} + d_{text} = 786 + 512 = 1298$.

### 2.3.2 Late Fusion (Neural Network)

Late fusion processes each modality through separate encoder networks before combining representations at a deeper layer. This allows modality-specific feature learning while enabling multimodal interactions in the fusion layer.

**Architecture**:
- Audio encoder: Linear(786→128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128→64) → BatchNorm → ReLU → Dropout(0.2)
- Text encoder: Linear(512→256) → BatchNorm → ReLU → Dropout(0.4) → Linear(256→64) → BatchNorm → ReLU → Dropout(0.2)
- Fusion classifier: Linear(128→64) → BatchNorm → ReLU → Dropout(0.2) → Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1)

The fusion model outputs logits that are converted to probabilities via sigmoid activation during inference.

## 2.4 Model Training

### 2.4.1 XGBoost (Early Fusion)

For the XGBoost classifier, we used the following hyperparameters:
- Number of estimators: 100
- Maximum depth: 5
- Learning rate: 0.1
- Subsample ratio: 0.8
- Column sample by tree: 0.8
- Scale positive weight: $n_{negative}/n_{positive}$ (approximately 2.3)

Hyperparameters were selected based on 5-fold cross-validation on the training set.

### 2.4.2 Neural Network (Late Fusion)

Training the late fusion model required careful regularization to prevent overfitting given limited data:

**Loss Function**: Binary cross-entropy with logits, weighted by class frequency:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ w \cdot y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right]$$
where $w = n_{negative}/n_{positive}$ balances the contribution of positive samples.

**Optimizer**: AdamW with learning rate $10^{-3}$ and weight decay $10^{-2}$ (L2 regularization).

**Learning Rate Schedule**: ReduceLROnPlateau with patience 5, reducing learning rate by factor 0.5 when F1 score plateaus.

**Regularization**:
- Batch normalization after each linear layer
- Dropout (0.2-0.4) for stochastic regularization
- Gradient clipping with max norm 1.0
- Early stopping with patience 20, monitoring validation F1 score

**Training Procedure**: Models were trained for up to 100 epochs with mini-batch size 16. We saved the model checkpoint achieving highest validation F1 score.

## 2.5 Model Calibration

Clinical deployment requires well-calibrated probability estimates, where predicted probabilities reflect true outcome frequencies. We applied Platt scaling (sigmoid calibration) to post-hoc calibrate model predictions.

Given uncalibrated logits $z$, calibrated probabilities are computed as:
$$p_{cal} = \sigma(Az + B)$$
where parameters $A$ and $B$ are learned on a held-out calibration set by minimizing negative log-likelihood.

We evaluated calibration quality using:
- **Expected Calibration Error (ECE)**: Average absolute difference between predicted probability and empirical accuracy across probability bins
- **Brier Score**: Mean squared error between predicted probabilities and binary outcomes

## 2.6 Evaluation Metrics

We report comprehensive metrics to assess different aspects of model performance:

**Discrimination Metrics**:
- Area Under ROC Curve (AUROC): Probability that a randomly chosen positive sample ranks higher than a randomly chosen negative sample
- Area Under Precision-Recall Curve (AUPRC): Summarizes precision-recall trade-off, particularly informative for imbalanced datasets
- Matthews Correlation Coefficient (MCC): Correlation between predicted and true labels, robust to class imbalance

**Classification Metrics** (at threshold 0.5):
- F1 Score: Harmonic mean of precision and recall
- Sensitivity (Recall): True positive rate
- Specificity: True negative rate, critical for clinical applications to minimize false positives
- Accuracy: Overall correct classification rate

**Calibration Metrics**:
- Expected Calibration Error (ECE): Lower values indicate better calibration
- Brier Score: Combines discrimination and calibration

### 2.6.1 Statistical Analysis

To quantify uncertainty in performance estimates, we computed 95% bootstrap confidence intervals with 1000 resamples for all metrics. We report point estimates with confidence intervals in the format: estimate [lower, upper].

## 2.7 Ablation Study

To understand the contribution of each modality and feature set, we conducted systematic ablation experiments testing all combinations:

1. Audio only (Wav2Vec + prosody)
2. Text only (BERT embeddings)
3. Audio only (prosody features only)
4. Audio only (Wav2Vec only)
5. Audio + Text (early fusion)
6. Audio + Text (late fusion)

For each configuration, we performed 5-fold stratified cross-validation and reported mean ± standard deviation across folds. This ablation reveals:
- Whether multimodal fusion provides benefit over unimodal approaches
- The relative contribution of pretrained embeddings versus hand-crafted features
- Whether modalities provide complementary or redundant information

## 2.8 Implementation

All experiments were implemented in Python 3.10 using:
- PyTorch 2.0 for neural network models
- XGBoost 1.7 for gradient boosting
- Transformers 4.30 for Wav2Vec 2.0
- SentenceTransformers 2.2 for text encoding
- Librosa 0.10 for audio feature extraction
- Scikit-learn 1.2 for preprocessing, calibration, and evaluation

Code is available at [repository URL].

## 2.9 Ethical Considerations

The DAIC-WOZ dataset was collected under IRB approval and participants provided informed consent. As the data is de-identified and publicly available for research purposes, no additional ethical approval was required for this study. We note that automated depression detection systems should be used as screening aids rather than diagnostic tools, with final clinical decisions made by qualified healthcare professionals.
