# Brain-Tumor-Classification-using-SoLA-Vision-Hybrid-Attention-
Brain Tumor Classification using SoLA-Vision, a custom hybrid attention architecture that combines Layer-wise Linear Attention and Softmax Attention for efficient multi-class MRI image classification using PyTorch.
Overview

This project presents SoLA-Vision, a custom transformer-inspired deep learning architecture for brain tumor classification using MRI images. The proposed model introduces a Layer-wise Hybrid Attention mechanism that combines Linear Attention and Softmax Attention to balance computational efficiency with global feature learning.

Unlike conventional Vision Transformers that apply softmax attention in every transformer block, SoLA-Vision follows a hybrid attention pattern:
L → L → L → S → L → L
where:

* L = Linear Attention
* S = Softmax Attention

This design reduces computational complexity while preserving the ability to capture long-range dependencies within MRI images.

⸻

Features

* Custom SoLA-Vision architecture
* Hybrid Linear + Softmax Attention
* Multi-class brain MRI classification
* Transformer-inspired model built from scratch
Preprocessing

* Duplicate removal
* Missing value check
* Train-validation split
* Oversampling for class balancing
* Image resizing (224 × 224)
* Data augmentation
* Normalization
Architecture
MRI Image
      ↓
Patch Embedding
      ↓
CLS Token + Positional Encoding
      ↓
Linear Attention
      ↓
Linear Attention
      ↓
Linear Attention
      ↓
Softmax Attention
      ↓
Linear Attention
      ↓
Linear Attention
      ↓
Classification Head
      ↓
Prediction

@ Muhammad Umar Nadeem, Brain Tumor Classification using Layer-wise Linear & Softmax Hybrid Attention (SoLA-Vision), 2025.
