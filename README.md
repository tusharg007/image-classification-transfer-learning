# Image Classification using Transfer Learning (ResNet50)

This project implements an image classification pipeline using convolutional neural networks (CNNs) and transfer learning with a pre-trained ResNet50 model. The goal is to evaluate how fine-tuning improves model performance while maintaining generalization.

## Dataset
- Cats vs Dogs dataset  
- Loaded using **TensorFlow Datasets (TFDS)**  
- Binary classification problem

## Tech Stack
- Python  
- TensorFlow / Keras  
- TensorFlow Datasets  
- Google Colab (GPU)

## Approach
1. Loaded a pre-trained **ResNet50** model trained on ImageNet.
2. Performed **feature extraction** by freezing the convolutional backbone.
3. Added a custom classification head using global average pooling and dense layers.
4. Applied **data augmentation** (random flip, rotation, zoom) to improve robustness.
5. Fine-tuned the top convolutional layers using a reduced learning rate to improve validation performance.

## Results
- Training accuracy improved steadily during fine-tuning.
- Validation accuracy reached approximately **75–78%**.
- Validation and training curves remained close, indicating **no significant overfitting**.
- Fine-tuning led to better generalization compared to feature extraction alone.

## Observations
- Initial validation fluctuations were observed due to frozen backbone and stochastic batches.
- Fine-tuning higher layers improved stability and overall accuracy.
- Data augmentation helped reduce overfitting.

## Notes
All experiments were conducted in **Google Colab** to leverage
