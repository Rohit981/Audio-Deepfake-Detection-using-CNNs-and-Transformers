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
