# Audio Deepfake Detection using CNNs and Transformers

A PyTorch implementation comparing convolutional neural networks and transformer-based architectures for audio deepfake detection using Mel spectrogram classification.

This project evaluates seven deep learning architectures under a unified training pipeline and investigates optimization strategies including architecture-specific fine-tuning, layer-wise learning rate decay, cosine annealing warm restarts, gradient clipping, and Equal Error Rate (EER)-driven evaluation.

## Features

- Mel Spectrogram preprocessing with offline feature caching
- RAM-cached dataset loading for fast training
- ResNet18
- ResNet50
- CNN-Transformer Hybrid
- Vision Transformer (ViT)
- DeiT (Hard & Soft Distillation)
- Custom Swin Transformer
- Layer-wise Learning Rate Decay (LLRD)
- Cosine Annealing Warm Restarts
- AdamW optimizer
- Gradient Clipping
- Checkpoint System
- Early Stopping
- ROC Curve
- Equal Error Rate (EER)
- Confusion Matrix
- F1 Score
- Recall

## Dataset

Dataset:
ASVspoof2019 Logical Access (LA)

The dataset is not included in this repository because of licensing restrictions.

Directory structure:

Datasets/
    ASVspoof2019_LA_train/
    ASVspoof2019_LA_dev/
    ASVspoof2019_LA_eval/

## 📚 Implemented Architectures

| Architecture | Description | Status |
|:-------------|:------------|:------:|
| **ResNet18** | Residual Convolutional Neural Network | ✅ |
| **ResNet50** | Deep Residual Convolutional Neural Network | ✅ |
| **CNN-Transformer** | CNN feature extractor with Transformer encoder | ✅ |
| **Vision Transformer (ViT)** | Pure Transformer-based image classifier | ✅ |
| **DeiT-Hard** | Data-efficient Image Transformer (Hard Distillation) | ✅ |
| **DeiT-Soft** | Data-efficient Image Transformer (Soft Distillation) | ✅ |
| **Swin Transformer** | Hierarchical Window-based Vision Transformer | ✅ |

## 📂 Data Pipeline

```text
ASVspoof2019 Audio (.flac)
            │
            ▼
      Librosa Loader
            │
            ▼
   Mel Spectrogram (128 bins)
            │
            ▼
      Power-to-dB
            │
            ▼
 Pad / Crop to 128 × 128
            │
            ▼
 Save as NumPy (.npy)
            │
            ▼
      RAM Feature Cache
```

## 🚀 Training Pipeline

```text
RAM Cached Features
        │
        ▼
 PyTorch DataLoader
        │
        ▼
 Data Augmentation
 (Gaussian Noise)
        │
        ▼
      Model
        │
        ▼
 BCEWithLogitsLoss
        │
        ▼
 AdamW Optimizer
        │
        ▼
 Gradient Clipping
        │
        ▼
 Cosine Annealing
 Warm Restarts
        │
        ▼
 Checkpointing
        │
        ▼
 Evaluation
```
## Experimental Pipeline

Phase I
- Unified baseline training

Phase II
- Architecture comparison

Phase III
- Architecture-specific fine-tuning
- Layer-wise learning rate decay
- Cosine Annealing Warm Restarts
- AdamW
- Gradient Clipping
- EER-based checkpoint selection

## 📊 Results

The final models were evaluated on the ASVspoof2019 Logical Access (LA) evaluation set using Test Accuracy, ROC-AUC, and Equal Error Rate (EER). Models are ranked by EER, with lower values indicating better detection performance.

| Rank | Model | Test Accuracy | ROC-AUC | EER ↓ |
|:---:|:--------------------------|:------------:|:-------:|:------:|
| 🥇 | **Swin Transformer 8H** | **91.95%** | 0.9844 | **5.86%** |
| 🥈 | **Swin Transformer 4H** | 91.84% | 0.9826 | 5.97% |
| 🥉 | **ResNet18** | 82.23% | 0.9864 | 6.01% |
| 4 | **ResNet50** | 84.55% | **0.9873** | 6.47% |
| 5 | **CNN-Transformer** | 57.40% | 0.9800 | 6.70% |
| 6 | **DeiT-Soft** | 90.50% | 0.9726 | 8.21% |
| 7 | **DeiT-Hard** | 80.07% | 0.9718 | 8.76% |
| 8 | **Vision Transformer (ViT)** | 81.68% | 0.9585 | 10.39% |

> **Note:** Results correspond to the reported fine-tuned evaluation runs. The leaderboard is ranked by EER, while ROC-AUC measures score-ranking discrimination and accuracy depends on the applied decision threshold. These metrics should be interpreted together, particularly where substantial discrepancies occur.

## Key Findings

- Swin Transformer 8H achieved the lowest EER: The eight-head configuration recorded an EER of 5.86%, a test accuracy of 91.95%, and a ROC-AUC of 0.9844, producing the strongest overall result by EER.

- Swin Transformer 4H demonstrated comparable performance: With an EER of 5.97% and an accuracy of 91.84%, the four-head configuration performed almost identically to the eight-head variant under the evaluated conditions.

- ResNet architectures remained highly competitive: ResNet18 achieved an EER of 6.01%, while ResNet50 achieved the highest ROC-AUC of 0.9873. This demonstrates that convolutional architectures remain strong baselines for spectrogram-based audio deepfake detection.

- Fine-tuning changed the architectural rankings: The Swin Transformer variants moved ahead of the residual networks in EER, illustrating that the relative performance of the architectures depended on the training and fine-tuning configuration.

- Accuracy and ranking metrics must be interpreted together: The CNN–Transformer achieved an EER of 6.70% and a ROC-AUC of 0.9800, but its test accuracy was only 57.40%. This discrepancy warrants further verification of the checkpoint, score convention, and decision threshold before its operational performance can be established.

- Metric validation was essential: Investigation of anomalous checkpoints highlighted the importance of examining ROC-AUC, EER, confusion matrices, score distributions, and model activations together rather than relying on a single reported metric.
  
Overall, the fine-tuned results identify the Swin Transformer 8H as the leading architecture by EER, with Swin Transformer 4H and ResNet18 close behind. The results also demonstrate that CNNs remain competitive and that robust model selection requires careful validation of both threshold-independent discrimination and threshold-dependent classification performance.
## Future Work

- BEATs
- AST (Audio Spectrogram Transformer)
- Self-supervised Audio Transformers
