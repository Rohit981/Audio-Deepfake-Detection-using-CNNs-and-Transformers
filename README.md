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
> # 🌐 Cross-Dataset Performance Comparison

The following table compares all eight architectures across ASVspoof2019 LA, ASVspoof2021 DF, WaveFake (LJSpeech–MelGAN), and In-the-Wild.

Each cell reports **Test Accuracy / ROC-AUC / EER**, respectively. Models are evaluated using their reported fine-tuned checkpoints. Lower EER and higher ROC-AUC generally indicate better discrimination.

| Model | ASVspoof2019 LA<br>Acc / AUC / EER | ASVspoof2021 DF<br>Acc / AUC / EER | WaveFake<br>Acc / AUC / EER | In-the-Wild<br>Acc / AUC / EER |
|:---|:---:|:---:|:---:|:---:|
| **Swin Transformer 8H** | 91.95% / 0.9844 / **5.86%** | 39.16% / 0.8883 / 20.75% | 50.01% / 0.7365 / **31.63%** | 38.01% / 0.4459 / 54.70% |
| **Swin Transformer 4H** | 91.84% / 0.9826 / 5.97% | 44.09% / 0.8876 / 20.83% | 50.01% / 0.7008 / 34.25% | 39.07% / 0.4367 / 54.85% |
| **ResNet18** | 82.23% / 0.9864 / 6.01% | 78.16% / 0.7609 / 32.93% | 50.02% / 0.4808 / 51.51% | 37.07% / 0.5438 / **47.36%** |
| **ResNet50** | 84.55% / 0.9873 / 6.47% | 78.19% / 0.8497 / 24.70% | 50.00% / 0.4486 / 53.75% | 36.97% / 0.4544 / 52.02% |
| **CNN-Transformer** | 57.40% / 0.9800 / 6.70% | 68.40% / 0.8792 / **20.70%** | 50.00% / 0.6735 / 37.54% | 35.70% / 0.5175 / **47.36%** |
| **DeiT-Soft** | 90.50% / 0.9726 / 8.21% | 71.25% / 0.8327 / 24.92% | 50.59% / 0.5981 / 43.37% | 35.69% / 0.3997 / 56.69% |
| **DeiT-Hard** | 80.07% / 0.9718 / 8.76% | 75.02% / 0.8369 / 24.30% | 50.48% / 0.5729 / 45.01% | 37.34% / 0.4183 / 55.48% |
| **Vision Transformer (ViT)** | 81.68% / 0.9585 / 10.39% | 66.73% / 0.8338 / 24.09% | 52.20% / 0.5827 / 44.23% | 34.86% / 0.3159 / 63.86% |

> **Note:** Accuracy is measured at the applied decision threshold, whereas ROC-AUC measures ranking performance and EER measures the operating point where false acceptance and false rejection rates are equal. Near-50% accuracy on a balanced dataset does not necessarily imply chance-level ranking performance. ROC-AUC values below 0.5 should be interpreted using the established class-label and score-polarity convention.

## Key Findings

- Swin Transformer 8H demonstrated the strongest overall cross-dataset consistency: It achieved the lowest EER on ASVspoof2019 LA (5.86%) and WaveFake (31.63%), while recording the highest ROC-AUC on ASVspoof2021 DF (0.8883). However, its performance deteriorated substantially on In-the-Wild, where its EER increased to 54.70%, demonstrating that its advantages did not transfer uniformly across datasets.

- Swin Transformer 4H produced results comparable to the eight-head configuration: It achieved an EER of 5.97% on ASVspoof2019 LA and 20.83% on ASVspoof2021 DF, placing it close to Swin Transformer 8H on both datasets. Its WaveFake EER of 34.25% was the second-lowest among the evaluated models, although its In-the-Wild EER increased to 54.85%.

- CNN–Transformer performed particularly well on ASVspoof2021 DF: It achieved the lowest EER of 20.70%, narrowly outperforming both Swin Transformer variants. It also tied with ResNet18 for the lowest reported EER on In-the-Wild at 47.36%. However, its low evaluation accuracy on ASVspoof2019 LA (57.40%) and In-the-Wild (35.70%) highlights the importance of distinguishing ranking performance from threshold-dependent classification accuracy.

- ResNet18 and ResNet50 remained competitive on the source dataset but generalized differently: ResNet18 achieved an EER of 6.01% on ASVspoof2019 LA and the highest In-the-Wild ROC-AUC among the evaluated architectures (0.5438), although this still indicated weak discrimination. ResNet50 achieved the highest source-domain ROC-AUC of 0.9873 and an EER of 24.70% on ASVspoof2021 DF, but its performance deteriorated considerably on WaveFake (53.75% EER) and In-the-Wild (52.02% EER).

- ViT and the DeiT variants showed weaker or less consistent generalization: ViT recorded EERs of 24.09% on ASVspoof2021 DF, 44.23% on WaveFake, and 63.86% on In-the-Wild. DeiT-Hard and DeiT-Soft achieved moderate results on ASVspoof2021 DF and WaveFake but had In-the-Wild EERs of 55.48% and 56.69%, respectively. These results suggest that the evaluated global-attention and distillation configurations did not provide a consistent advantage under cross-dataset conditions.

- External evaluation exposed a substantial generalization gap: The best EER increased from 5.86% on ASVspoof2019 LA to 20.70% on ASVspoof2021 DF, 31.63% on WaveFake, and 47.36% on In-the-Wild. The corresponding best ROC-AUC decreased from 0.9873 on ASVspoof2019 LA to 0.8883 on ASVspoof2021 DF, 0.7365 on WaveFake, and 0.5438 on In-the-Wild. This decline demonstrates that high source-domain performance did not guarantee reliable discrimination on unseen datasets.

- Accuracy, ROC-AUC, and EER must be interpreted together: Several models recorded evaluation accuracies near 50% on WaveFake despite ROC-AUC values above chance, while the Swin Transformer variants showed similarly large discrepancies between accuracy and ROC-AUC on ASVspoof2021 DF. These results highlight the importance of examining decision-threshold transfer alongside ranking discrimination and EER rather than relying on any single metric.
  
Overall, Swin Transformer 8H was the strongest model by EER on ASVspoof2019 LA and WaveFake, while CNN–Transformer achieved the lowest EER on ASVspoof2021 DF and tied with ResNet18 on In-the-Wild. No architecture consistently dominated all four datasets. The findings emphasize that architecture selection should consider both source-domain performance and robustness to unseen synthesis systems and recording conditions. They also motivate further investigation into score calibration, preprocessing sensitivity, and potential reliance on dataset-specific acoustic features.

## Future Work

- BEATs
- AST (Audio Spectrogram Transformer)
- Self-supervised Audio Transformers
