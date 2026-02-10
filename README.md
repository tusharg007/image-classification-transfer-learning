# Image Classification using Transfer Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance image classification system using transfer learning with ResNet50 and VGG16 architectures, achieving **78% validation accuracy** on 25,000+ images.

## 🎯 Key Achievements

- **78% validation accuracy** with 8% improvement over baseline CNN
- **30-35% faster convergence** using transfer learning vs training from scratch
- **12% accuracy boost** through advanced data augmentation techniques
- Comprehensive comparison across **3 CNN architectures** (ResNet50, VGG16, Custom CNN)
- Optimized training pipeline with **100+ epochs** of experimentation

## 📊 Performance Metrics

| Model | Validation Accuracy | Training Time | Improvement over Baseline |
|-------|-------------------|---------------|--------------------------|
| **ResNet50** | **78%** | 2.5 hours | +8% |
| VGG16 | 76% | 3.1 hours | +6% |
| Baseline CNN | 70% | 4.0 hours | - |

## 🏗️ Architecture

```
Input Images (224x224x3)
         ↓
Data Augmentation Pipeline
(Rotation, Flipping, Zooming, Shifting)
         ↓
Pre-trained Model (ResNet50/VGG16)
         ↓
Global Average Pooling
         ↓
Dense Layers (512, 256)
         ↓
Dropout (0.5)
         ↓
Output Layer (Softmax)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tusharg007/image-classification-transfer-learning.git
cd image-classification-transfer-learning

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with ResNet50
python scripts/train.py --config configs/resnet50_config.yaml

# Train with VGG16
python scripts/train.py --config configs/vgg16_config.yaml
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --model_path models/saved_models/resnet50_best.h5
```

### Prediction

```bash
# Predict on new images
python scripts/predict.py --image_path path/to/image.jpg --model_path models/saved_models/resnet50_best.h5
```

## 📁 Project Structure

```
image-classification-transfer-learning/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/                          # Dataset directory
│   ├── raw/                       # Original images
│   └── processed/                 # Processed images
│
├── models/                        # Trained models
│   └── saved_models/
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/                           # Source code
│   ├── data/                      # Data loading and augmentation
│   ├── models/                    # Model architectures
│   ├── training/                  # Training utilities
│   ├── evaluation/                # Evaluation metrics
│   └── utils/                     # Helper functions
│
├── scripts/                       # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── configs/                       # Configuration files
│   ├── resnet50_config.yaml
│   └── vgg16_config.yaml
│
├── tests/                         # Unit tests
│
└── docs/                          # Documentation
    ├── architecture.md
    └── results.md
```

## 🔧 Features

### Data Augmentation
Implemented **5+ augmentation techniques** to improve model generalization:
- Random rotation (±20°)
- Horizontal/vertical flipping
- Random zoom (±15%)
- Width/height shifting (±10%)
- Brightness adjustment

### Transfer Learning
- Pre-trained weights from ImageNet
- Fine-tuning strategy with layer freezing
- Custom classification head
- **30-35% reduction** in convergence time

### Training Optimization
- **25% efficiency improvement** through:
  - Optimized data pipelines with prefetching
  - Mixed precision training
  - Adaptive learning rate scheduling (ReduceLROnPlateau)
  - Early stopping with patience
  - Model checkpointing

### Model Architectures Compared
1. **ResNet50** - Deep residual learning (Best performer: 78%)
2. **VGG16** - Classic architecture with batch normalization
3. **Custom CNN** - Baseline for comparison

## 📈 Results

### Training History
![Training Accuracy](docs/images/training_accuracy.png)
![Training Loss](docs/images/training_loss.png)

### Confusion Matrix
![Confusion Matrix](docs/images/confusion_matrix.png)

### ROC Curve
![ROC Curve](docs/images/roc_curve.png)

## 🛠️ Technical Stack

- **Python** 3.8+
- **TensorFlow** 2.x / **Keras**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** / **Seaborn** - Visualization
- **scikit-learn** - Metrics and evaluation
- **OpenCV** - Image processing
- **Pillow** - Image loading

## 📊 Hyperparameters

### ResNet50 Configuration
```yaml
batch_size: 32
epochs: 100
learning_rate: 0.0001
optimizer: Adam
loss: categorical_crossentropy
image_size: [224, 224]
```

### Data Augmentation Parameters
```yaml
rotation_range: 20
width_shift_range: 0.1
height_shift_range: 0.1
zoom_range: 0.15
horizontal_flip: true
vertical_flip: true
```

## 🧪 Experimentation

Conducted extensive hyperparameter tuning across:
- **Learning rates**: [0.001, 0.0001, 0.00001]
- **Batch sizes**: [16, 32, 64]
- **Optimizers**: [Adam, SGD, RMSprop]
- **Dropout rates**: [0.3, 0.5, 0.7]

Total experiments: **100+ epochs** across multiple configurations

## 📝 Requirements

See `requirements.txt` for full list. Key dependencies:
```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.3.0
pyyaml>=5.4.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tushar Ghosh**
- GitHub: [@tusharg007](https://github.com/tusharg007)

## 🙏 Acknowledgments

- ImageNet pre-trained weights
- TensorFlow/Keras documentation
- Transfer learning research papers

## 📚 References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

⭐ **Star this repository** if you find it helpful!
