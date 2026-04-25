# Lizard Species Classification - Three-Notebook Pipeline

## Overview
This project is split into **3 modular notebooks** optimized for maximum Kaggle accuracy while maintaining reproducibility and team defensibility.

**Key Principle**: No data is deleted - only valid/loadable images are used for training, maximizing available information.

---

## Notebook Execution Order

### 1. **EDA and Dataset Preparation** (`01_EDA_and_Dataset_Preparation.ipynb`)
**Purpose**: Comprehensive exploratory analysis, data quality checks, and metadata extraction

**What it does**:
- ✅ Loads and validates all CSVs (train.csv, test.csv, sample_submission.csv)
- ✅ Resolves underscore/hyphen mismatch between folder names and filenames
- ✅ Checks for corrupted/missing images (no deletion - just filtering)
- ✅ Analyzes class distribution and identifies imbalance (18% ratio)
- ✅ Collects image metadata (dimensions, formats, aspect ratios)
- ✅ Creates visualizations of class distribution and sample images
- ✅ Saves clean datasets and metadata to `metadata/` directory

**Outputs to `metadata/` folder**:
- `clean_train.csv` - Valid training samples only
- `clean_test.csv` - Valid test samples only
- `class_mappings.json` - Class name ↔ index mappings & weights
- `eda_config.json` - Dataset statistics and config
- `train_image_metadata.csv` - Image size/format data
- `test_image_metadata.csv` - Test image metadata
- Visualization PNG files

**Run time**: ~5-10 minutes (depending on system)

---

### 2. **Model Training** (`02_Model_Training.ipynb`)
**Purpose**: Build, train, and optimize transfer learning model for best accuracy

**What it does**:
- ✅ Loads clean data and metadata from EDA notebook
- ✅ Creates stratified 85/15 train/validation split (preserves class distribution)
- ✅ Builds data pipelines with **strong augmentation** for training:
  - Random flip (H/V), rotation (±20°), zoom (0.75-1.3x)
  - Brightness and contrast adjustments
  - Applied only to training set (validation/test deterministic)
- ✅ **Stage 1**: Trains with frozen backbone (quick baseline, 30 epochs)
- ✅ **Stage 2**: Fine-tunes upper 40% of backbone (20 epochs)
- ✅ **Stage 3**: Aggressive fine-tuning of upper 70% of backbone (10 epochs)
- ✅ Uses class weights to handle imbalance
- ✅ Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

**Model Architecture**:
- Backbone: **EfficientNetB2** (ImageNet pre-trained, 224×224 input)
- Head: GAP → Dense(512) + BatchNorm → Dropout(0.4) → Dense(256) + BatchNorm → Dropout(0.4) → Dense(7, softmax)
- L2 regularization (1e-4) throughout

**Configuration (easily tunable)**:
- Image size: 224×224
- Batch size: 32
- Augmentation strength: Strong
- Dropout: 0.4
- L2 regularization: 1e-4
- Class weighting: Enabled

**Outputs to `models/` folder**:
- `final_model.h5` - Best trained model (HDF5 format)
- `final_model_savedmodel/` - Model in SavedModel format
- `training_history.png` - Loss/accuracy across all 3 stages
- `training_results.json` - Metrics and configuration
- Stage 1/2/3 checkpoint models (if you want to revert)

**Run time**: ~60-90 minutes on consumer GPU (RTX 30/40 series)

---

### 3. **Model Evaluation and Testing** (`03_Model_Evaluation_and_Testing.ipynb`)
**Purpose**: Comprehensive evaluation and Kaggle submission generation

**What it does**:
- ✅ Loads trained model and metadata
- ✅ Evaluates on validation set
- ✅ Generates predictions on all 325 test images
- ✅ Computes confusion matrix (7×7) with analysis
- ✅ Identifies top misclassification pairs
- ✅ Analyzes prediction confidence scores
- ✅ Visualizes misclassified validation examples
- ✅ Creates Kaggle submission file

**Key Metrics Generated**:
- Overall validation accuracy
- Per-class accuracy (identifies strengths/weaknesses)
- Confusion matrix (raw + normalized)
- Top 5 confusion pairs (for team discussion)
- Prediction confidence analysis
- Per-class performance charts

**Outputs to `results/` folder**:
- **`submission.csv`** ← **KAGGLE SUBMISSION FILE**
- `evaluation_report.json` - Detailed metrics
- `confusion_matrix.png` - Visual confusion matrix
- `misclassified_examples.png` - Failed predictions with explanations
- `confidence_distribution.png` - Confidence score histograms
- `per_class_accuracy.png` - Bar chart of class-wise performance

**Also copies**:
- `submission.csv` to main directory for easy access

**Run time**: ~10-20 minutes

---

## Quick Start Guide

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn pillow matplotlib seaborn scipy
```

### Execution
```
1. Open and run: 01_EDA_and_Dataset_Preparation.ipynb
   (Creates metadata/ folder with clean datasets)

2. Open and run: 02_Model_Training.ipynb
   (Creates models/ folder with trained model)

3. Open and run: 03_Model_Evaluation_and_Testing.ipynb
   (Creates results/ folder and submission.csv for Kaggle)
```

### For Highest Kaggle Score (if time permits)
After stage 3 in notebook 2, you can:
- Re-run notebook 3 to get fresh test predictions
- Or experiment with different backbones:
  - **EfficientNetB3** (larger, slower, potentially better accuracy)
  - **EfficientNetV2B0/B1** (newer architecture)
  - Ensemble multiple models and average predictions

---

## Key Design Decisions for Maximum Accuracy

### 1. **No Data Deletion**
- Only 1-3 images per 1334 may be unloadable
- We filter but keep all valid data (~99.8% retention)
- More data = higher accuracy potential

### 2. **Three-Stage Fine-tuning**
- **Stage 1**: Frozen backbone learns task quickly (warm-up)
- **Stage 2**: Unfreeze top 40% for moderate adaptation
- **Stage 3**: Unfreeze top 70% for maximum domain fit (late-stage aggressive tuning)
- Each stage uses lower learning rate to prevent catastrophic forgetting

### 3. **Strong Data Augmentation**
- Aggressive augmentation (flip, rotate±20°, zoom 0.75-1.3x) on training only
- Increases effective dataset size from 1,143 → ~3,000+ unique variations
- Validation/test remain deterministic for fair evaluation

### 4. **Class Weighting**
- Dataset has 18% class imbalance (178-199 samples per class)
- Loss weighting prevents bias toward majority classes
- Minority classes get equal learning priority

### 5. **Early Stopping + Learning Rate Scheduling**
- EarlyStopping prevents overfitting and wasted compute
- ReduceLROnPlateau fine-tunes when progress stalls
- Callbacks save best weights automatically

### 6. **Larger Backbone**
- EfficientNetB2 (vs B0) for better accuracy
- Still fits on consumer GPU (32 batch size optimal)
- ~9M parameters total (manageable for 1.1K training samples)

---

## Troubleshooting

### Out of Memory?
- Reduce `batch_size` to 16 in CONFIG (notebook 2)
- Or use smaller backbone: `EfficientNetB0` or `EfficientNetB1`

### Training too slow?
- Reduce `epochs_stage1/2/3` in CONFIG
- Or reduce `image_size` to 192 (may hurt accuracy slightly)

### Validation accuracy plateauing?
- Increase `augmentation_strength` to "strong" (already default)
- Increase dropout: `'dropout_rate': 0.5`
- Try EfficientNetB3 backbone

### Bad accuracy on certain classes?
- Check `confusion_matrix.png` from notebook 3
- Increase class weight for minority classes in notebook 2
- Manually inspect misclassified examples

---

## Defense Talking Points (for Kaggle competition)

1. **Why three notebooks?**
   - Modularity: Each step is clear and reproducible
   - Debuggability: Easy to fix issues without re-running all
   - Transparency: Team can understand and defend each phase

2. **Why no data deletion?**
   - Maximize signal: 1,334 labeled images every one counts
   - Ethical: Don't throw away training data
   - Practical: 99.8% retention rate is excellent

3. **Why EfficientNetB2?**
   - Best efficiency/accuracy trade-off for consumer GPU
   - 9M parameters (not too large to overfit on 1.1K samples)
   - Well-established in transfer learning literature

4. **Why three training stages?**
   - Stage 1: Warm-up (learn task from scratch with frozen features)
   - Stage 2: Balanced (adapt top features without destroying base)
   - Stage 3: Aggressive (final push for maximum accuracy)
   - Prevents catastrophic forgetting common in naive fine-tuning

5. **Why strong augmentation?**
   - Dataset is small (1.1K train samples)
   - Augmentation creates 3K+ unique variations
   - Generalization improves significantly

---

## File Structure

```
lizardwizard/
├── 01_EDA_and_Dataset_Preparation.ipynb
├── 02_Model_Training.ipynb
├── 03_Model_Evaluation_and_Testing.ipynb
├── README_NOTEBOOKS.md (this file)
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/ (class folders with images)
├── test/ (unlabeled test images)
├── metadata/ (created by notebook 1)
│   ├── clean_train.csv
│   ├── clean_test.csv
│   ├── class_mappings.json
│   ├── eda_config.json
│   └── *.png (EDA visualizations)
├── models/ (created by notebook 2)
│   ├── final_model.h5
│   ├── final_model_savedmodel/
│   ├── training_results.json
│   └── training_history.png
├── results/ (created by notebook 3)
│   ├── submission.csv ← KAGGLE SUBMISSION
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   ├── misclassified_examples.png
│   ├── confidence_distribution.png
│   └── per_class_accuracy.png
└── submission.csv (copy for convenience)
```

---

## Expected Results

Based on configuration and architecture:
- **Validation accuracy**: 85-92% (highly depends on augmentation seed)
- **Per-class accuracy**: 80-95% (class-dependent based on similarity)
- **Mean prediction confidence**: 0.88-0.95
- **Kaggle leaderboard**: Top 10-30% (competitive but not #1)

To push further:
- Ensemble 3-4 different backbones (B0, B2, V2B0) and average predictions
- Test-time augmentation (TTA): predict on 5 augmented versions and average
- Larger backbone (B3/B4) if GPU allows
- Manual hyperparameter search with Optuna

---

**Happy training! May your model reign supreme in the lizard kingdom!** 🦎
