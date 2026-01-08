# Image Classification using Transfer Learning (ResNet50)

This project implements an image classification pipeline using transfer learning with a pre-trained ResNet50 model.

## Dataset
Cats vs Dogs dataset loaded using TensorFlow Datasets (TFDS).

## Tech Stack
- Python
- TensorFlow / Keras
- Google Colab (GPU)

## Approach
- Used ResNet50 pre-trained on ImageNet
- Froze backbone layers for feature extraction
- Trained a custom classification head

## Results
Achieved ~85–90% validation accuracy within a few epochs.

## Notes
All experiments were conducted in Google Colab to avoid local hardware constraints.
