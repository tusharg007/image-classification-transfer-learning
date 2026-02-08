# Quick Start Guide

This guide will help you get up and running with the image classification project in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for faster training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tusharg007/image-classification-transfer-learning.git
cd image-classification-transfer-learning
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Directory Structure

Organize your dataset in the following structure:

```
data/
└── raw/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class_n/
        ├── image1.jpg
        └── ...
```

### Example: Using CIFAR-10

```python
import tensorflow as tf

# Download and prepare CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Save to directory structure
# (Implementation in notebooks/00_dataset_preparation.ipynb)
```

## Training Your First Model

### Option 1: Using Python Scripts (Recommended)

```bash
# Train ResNet50
python scripts/train.py --config configs/resnet50_config.yaml

# Train VGG16
python scripts/train.py --config configs/vgg16_config.yaml
```

### Option 2: Using Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_complete_pipeline.ipynb
# Run cells sequentially
```

### Option 3: Using Python API

```python
import tensorflow as tf
from src.data.data_loader import ImageDataLoader
from src.models.transfer_learning import get_model
from src.training.trainer import ModelTrainer

# Load data
loader = ImageDataLoader(batch_size=32)
train_ds, val_ds, num_classes = loader.load_data_from_directory('data/raw')

# Create model
model = get_model('resnet50', num_classes=num_classes)

# Train
trainer = ModelTrainer(model, 'resnet50')
trainer.compile_model(learning_rate=0.0001)
history = trainer.train(train_ds, val_ds, epochs=50)
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

### Training Logs

Monitor `logs/resnet50_training_log.csv` for metrics:
- epoch
- loss
- accuracy
- val_loss
- val_accuracy

## Model Evaluation

### Evaluate Saved Model

```bash
python scripts/evaluate.py \
    --model_path models/saved_models/resnet50_best.h5 \
    --data_dir data/test \
    --results_dir results
```

### Outputs

- `confusion_matrix.png` - Visual confusion matrix
- `roc_curve.png` - ROC curves for all classes
- `classification_report.csv` - Detailed metrics

## Making Predictions

### Single Image

```bash
python scripts/predict.py \
    --model_path models/saved_models/resnet50_best.h5 \
    --image_path path/to/image.jpg \
    --class_names dog cat bird
```

### Batch Prediction

```bash
python scripts/predict.py \
    --model_path models/saved_models/resnet50_best.h5 \
    --image_dir path/to/images/ \
    --save_results \
    --results_dir results/predictions
```

## Configuration

### Customizing Training

Edit `configs/resnet50_config.yaml`:

```yaml
# Model Configuration
model:
  name: "resnet50"
  num_classes: 10  # Change this

# Data Configuration
data:
  data_dir: "data/raw"
  batch_size: 32     # Adjust based on GPU memory
  validation_split: 0.2

# Training Configuration
training:
  epochs: 100        # Reduce for faster training
  initial_learning_rate: 0.0001
```

### Key Parameters to Adjust

**For Faster Training:**
```yaml
training:
  epochs: 50
data:
  batch_size: 64  # If you have enough GPU memory
```

**For Better Accuracy:**
```yaml
augmentation:
  enabled: true
  rotation_range: 30      # Increase augmentation
  zoom_range: 0.2
training:
  epochs: 150             # Train longer
fine_tuning:
  enabled: true          # Enable fine-tuning
```

**For Limited GPU Memory:**
```yaml
data:
  batch_size: 16         # Smaller batches
mixed_precision:
  enabled: false         # Disable mixed precision
```

## Common Issues

### Issue: Out of Memory (OOM)

**Solution:**
```yaml
# Reduce batch size in config
data:
  batch_size: 16  # or even 8
```

### Issue: Training Too Slow

**Solutions:**
1. Enable GPU if available
2. Reduce image size
3. Use mixed precision training
4. Reduce number of epochs

### Issue: Low Accuracy

**Solutions:**
1. Increase training epochs
2. Enable data augmentation
3. Try different learning rates
4. Use fine-tuning
5. Collect more training data

### Issue: Model Overfitting

**Solutions:**
```yaml
training:
  callbacks:
    early_stopping:
      patience: 10  # Stop earlier
augmentation:
  enabled: true   # Add augmentation
```

## Performance Benchmarks

### Expected Results

| Model | Dataset | Accuracy | Training Time (GPU) |
|-------|---------|----------|---------------------|
| ResNet50 | CIFAR-10 | 78% | 2-3 hours |
| VGG16 | CIFAR-10 | 76% | 3-4 hours |
| ResNet50 | Custom (10 classes) | 70-80% | 2-3 hours |

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB

**Recommended:**
- GPU: NVIDIA GTX 1060 or better
- RAM: 16GB
- Storage: 50GB SSD

## Next Steps

1. **Experiment with Hyperparameters**
   - Try different learning rates
   - Adjust batch sizes
   - Modify augmentation settings

2. **Try Different Architectures**
   ```bash
   python scripts/train.py --config configs/vgg16_config.yaml
   ```

3. **Fine-tune Models**
   ```yaml
   fine_tuning:
     enabled: true
     epochs: 50
   ```

4. **Deploy Your Model**
   - Convert to TensorFlow Lite
   - Export to ONNX
   - Deploy with TensorFlow Serving

## Additional Resources

- **Documentation:** See `docs/` directory
- **Examples:** See `notebooks/` directory
- **Architecture:** See `docs/architecture.md`
- **Contributing:** See `CONTRIBUTING.md`

## Getting Help

- **Issues:** Open an issue on GitHub
- **Discussions:** Use GitHub Discussions
- **Email:** Contact the maintainers

## Quick Reference

### Training Commands

```bash
# Basic training
python scripts/train.py --config configs/resnet50_config.yaml

# With GPU selection
python scripts/train.py --config configs/resnet50_config.yaml --gpu 0

# Evaluation
python scripts/evaluate.py --model_path models/saved_models/resnet50_best.h5 --data_dir data/test

# Prediction
python scripts/predict.py --model_path models/saved_models/resnet50_best.h5 --image_path test.jpg
```

### Python API

```python
# Quick training
from src.models.transfer_learning import get_model
from src.training.trainer import ModelTrainer

model = get_model('resnet50', num_classes=10)
trainer = ModelTrainer(model, 'my_model')
trainer.compile_model()
history = trainer.train(train_ds, val_ds, epochs=50)
```

---

**Happy Training! 🚀**

For detailed documentation, see the full [README.md](README.md) and [Architecture Guide](docs/architecture.md).
