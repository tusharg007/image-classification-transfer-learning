#!/usr/bin/env python3
"""
Training script for image classification with transfer learning.

Usage:
    python scripts/train.py --config configs/resnet50_config.yaml
"""

import argparse
import os
import sys
import yaml
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import ImageDataLoader, calculate_class_weights
from src.data.augmentation import DataAugmentation, apply_augmentation_to_dataset
from src.models.transfer_learning import get_model
from src.training.trainer import ModelTrainer, setup_mixed_precision
from src.evaluation.metrics import ModelEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device to use'
    )
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup mixed precision if enabled
    if config.get('mixed_precision', {}).get('enabled', False):
        setup_mixed_precision()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(config['data']['seed'])
    
    print("="*70)
    print("IMAGE CLASSIFICATION WITH TRANSFER LEARNING")
    print("="*70)
    print(f"Model: {config['model']['name'].upper()}")
    print(f"Dataset: {config['data']['data_dir']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Initial LR: {config['training']['initial_learning_rate']}")
    print("="*70 + "\n")
    
    # Initialize data loader
    print("Loading dataset...")
    data_loader = ImageDataLoader(
        image_size=tuple(config['model']['input_shape'][:2]),
        batch_size=config['data']['batch_size'],
        validation_split=config['data']['validation_split'],
        seed=config['data']['seed']
    )
    
    # Load data
    train_ds, val_ds, num_classes = data_loader.load_data_from_directory(
        data_dir=config['data']['data_dir'],
        color_mode=config['data']['color_mode'],
        shuffle=config['data']['shuffle']
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}\n")
    
    # Update num_classes in config
    config['model']['num_classes'] = num_classes
    
    # Setup data augmentation
    if config['augmentation']['enabled']:
        print("Setting up data augmentation...")
        augmentation = DataAugmentation(
            rotation_range=config['augmentation']['rotation_range'],
            width_shift_range=config['augmentation']['width_shift_range'],
            height_shift_range=config['augmentation']['height_shift_range'],
            zoom_range=config['augmentation']['zoom_range'],
            horizontal_flip=config['augmentation']['horizontal_flip'],
            vertical_flip=config['augmentation']['vertical_flip'],
            brightness_range=tuple(config['augmentation']['brightness_range']),
            fill_mode=config['augmentation']['fill_mode']
        )
        
        aug_model = augmentation.build_augmentation_model()
        train_ds = apply_augmentation_to_dataset(train_ds, aug_model)
        print("✓ Data augmentation enabled\n")
    
    # Create model
    print(f"Building {config['model']['name']} model...")
    model = get_model(
        model_name=config['model']['name'],
        input_shape=tuple(config['model']['input_shape']),
        num_classes=num_classes,
        base_trainable=config['model']['base_trainable']
    )
    
    print("✓ Model created successfully")
    print(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ])
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        model_name=config['model']['name'],
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['model_save_dir']
    )
    
    # Compile model
    print("Compiling model...")
    trainer.compile_model(
        learning_rate=config['training']['initial_learning_rate'],
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )
    print("✓ Model compiled\n")
    
    # Calculate class weights if needed
    class_weights = None
    # Uncomment to use class weights for imbalanced datasets
    # class_weights = calculate_class_weights(train_ds, num_classes)
    
    # Train model
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=config['training']['epochs'],
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Get best results
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Achieved at epoch: {best_epoch}")
    
    # Save final model
    trainer.save_model()
    
    # Fine-tuning (optional)
    if config.get('fine_tuning', {}).get('enabled', False):
        print("\n" + "="*70)
        print("STARTING FINE-TUNING")
        print("="*70 + "\n")
        
        history_fine = trainer.fine_tune(
            train_dataset=train_ds,
            val_dataset=val_ds,
            num_layers_to_unfreeze=config['fine_tuning']['unfreeze_layers'],
            fine_tune_epochs=config['fine_tuning']['epochs'],
            fine_tune_lr=config['fine_tuning']['learning_rate']
        )
        
        print("\n" + "="*70)
        print("FINE-TUNING COMPLETED")
        print("="*70)
        
        best_val_acc_fine = max(history_fine.history['val_accuracy'])
        print(f"\nBest Validation Accuracy (Fine-tuned): {best_val_acc_fine*100:.2f}%")
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("EVALUATION ON VALIDATION SET")
    print("="*70 + "\n")
    
    # Get class names from dataset
    class_names = train_ds.class_names if hasattr(train_ds, 'class_names') else \
                  [f"Class_{i}" for i in range(num_classes)]
    
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Get predictions
    y_true, y_pred, y_pred_proba = evaluator.get_predictions(model, val_ds)
    
    # Generate classification report
    evaluator.generate_classification_report(y_true, y_pred)
    
    # Calculate per-class accuracy
    evaluator.calculate_per_class_accuracy(y_true, y_pred)
    
    # Plot training history
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    evaluator.plot_training_history(
        history,
        save_path=os.path.join(
            config['paths']['results_dir'],
            f"{config['model']['name']}_training_history.png"
        )
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=os.path.join(
            config['paths']['results_dir'],
            f"{config['model']['name']}_confusion_matrix.png"
        )
    )
    
    # Plot ROC curve
    evaluator.plot_roc_curve(
        y_true,
        y_pred_proba,
        save_path=os.path.join(
            config['paths']['results_dir'],
            f"{config['model']['name']}_roc_curve.png"
        )
    )
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nModel saved to: {config['paths']['model_save_dir']}")
    print(f"Results saved to: {config['paths']['results_dir']}")
    print(f"Logs saved to: {config['paths']['log_dir']}")
    print("\n✓ Training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
