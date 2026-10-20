# Audio Deepfake Detection using CNNs and Transformers

A PyTorch implementation comparing convolutional neural networks, 
Transformer-based architectures, and hybrid models for audio deepfake 
detection using log-mel spectrogram classification.

This project evaluates eight model configurations on ASVspoof2019 
Logical Access (LA) and investigates cross-dataset generalization on 
ASVspoof2021 DF, In-the-Wild, and the LJSpeech–MelGAN subset of 
WaveFake.

The experiments compare ResNet18, ResNet50, a custom CNN–Transformer, 
Vision Transformer (ViT), DeiT-Hard, DeiT-Soft, and Swin Transformer 
configurations with four and eight attention heads.

The training pipeline incorporates architecture-specific fine-tuning, 
layer-wise learning-rate decay, gradient clipping, learning-rate 
scheduling, and EER-based checkpoint selection. External datasets are 
used to assess generalization beyond the source training domain.

## Features

- Mel Spectrogram preprocessing with offline feature caching
- RAM-cached dataset loading for fast training
- ResNet18
- ResNet50
- CNN-Transformer Hybrid
- Vision Transformer (ViT)
- DeiT (Hard & Soft Distillation)
- Custom Swin Transformer
- Architecture-specific fine-tuning
- Layer-wise Learning Rate Decay (LLRD)
- Linear Learning-Rate Warm-up
- Cosine Annealing Warm Restarts
- AdamW optimizer
- Gradient Clipping
- EER-based Checkpoint Selection
- Early Stopping
- ROC Curve
- Equal Error Rate (EER)
- Confusion Matrix
- F1 Score
- Recall

## Dataset

Dataset:
ASVspoof2019 Logical Access (LA), ASVspoof2021 DF, In-the-Wild, WaveFake

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
Linear LR Warm-up
      │
      ▼
Cosine Annealing
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
> # 🌐 Cross-Dataset Performance Comparison

The following table compares all eight architectures across ASVspoof2019 LA, ASVspoof2021 DF, WaveFake (LJSpeech–MelGAN), and In-the-Wild.

Each cell reports **EER**, respectively. Models are evaluated using their reported fine-tuned checkpoints. Lower EER and generally indicate better discrimination.

| Model | ASVspoof2019 LA | ASVspoof2021 DF | WaveFake | In-the-Wild |
|:---|:---:|:---:|:---:|:---:|
| **Swin Transformer 8H** | **5.86%** | 20.75% | **31.63%** | 54.70% |
| **Swin Transformer 4H** | 5.97% | 20.83% | 34.25% | 54.85% |
| **ResNet18** | 6.01% | 32.93% | 51.51% | **47.36%** |
| **ResNet50** | 6.47% | 24.70% | 53.75% | 52.02% |
| **CNN-Transformer** | 6.70% | **20.70%** | 37.54% | **47.36%** |
| **DeiT-Soft** | 8.21% | 24.92% | 43.37% | 56.69% |
| **DeiT-Hard** | 8.76% | 24.30% | 45.01% | 55.48% |
| **Vision Transformer (ViT)** | 10.39% | 24.09% | 44.23% | 63.86% |

> **Note:** Values represent Equal Error Rate (EER). Lower EER indicates better performance. Bold values highlight the lowest EER for each dataset.

## Key Findings

- **Strong source-domain performance:** Swin Transformer 8H achieved the lowest ASVspoof2019 LA EER of 5.86%, followed closely by Swin Transformer 4H and ResNet18.

- **Architecture rankings changed across datasets:** CNN–Transformer achieved the lowest EER on ASVspoof2021 DF, while Swin Transformer 8H  performed best on WaveFake.

- **Real-world generalization remained challenging:** In-the-Wild produced EERs of 47.36% or higher across the evaluated models, indicating limited discrimination under the reported evaluation conditions.

- **Convolutional models remained competitive:** ResNet18 and ResNet50 performed well on ASVspoof2019 LA, demonstrating that Transformer architectures do not consistently outperform CNNs.

- **Accuracy and ranking metrics differed:** Several models achieved ROC-AUC values above chance despite test accuracies near 50% on WaveFake. This highlights the importance of examining decision thresholds alongside ROC-AUC and EER.

- **Generalization remains the central challenge:** The substantial increase in EER across external datasets suggests that strong source-domain performance does not guarantee robust detection of unseen synthesis methods or audio sources.
  
Overall, Swin Transformer 8H was the strongest model by EER on ASVspoof2019 LA and WaveFake, while CNN–Transformer achieved the lowest EER on ASVspoof2021 DF and tied with ResNet18 on In-the-Wild. No architecture consistently dominated all four datasets. The findings emphasize that architecture selection should consider both source-domain performance and robustness to unseen synthesis systems and recording conditions. They also motivate further investigation into score calibration, preprocessing sensitivity, and potential reliance on dataset-specific acoustic features.

## Future Work

- **Self-Supervised Audio Representations:** Evaluate pretrained audio encoders such as BEATs and WavLM to investigate whether representations learned from broader audio data improve cross-dataset generalization.

- **Audio Spectrogram Transformer (AST):** Compare AST against the existing vision-based Transformer architectures using the same evaluation protocol.

- **Generalization Across Synthesis Systems:** Expand evaluation to additional vocoders and deepfake datasets to determine whether models learn transferable spoofing cues or rely on generator-specific characteristics.

- **Robust Preprocessing:** Investigate the effects of compression, duration, spectrogram padding, and other preprocessing choices.

- **Calibration and Threshold Transfer:** Examine how decision thresholds selected on source-domain validation data behave on external datasets.

- **Controlled Ablation Studies:** Evaluate the contributions of layer-wise learning-rate decay, data augmentation, and architecture-specific optimization under controlled experimental conditions.
