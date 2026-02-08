#!/usr/bin/env python3
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --model_path models/saved_models/resnet50_best.h5 --data_dir data/test
"""

import argparse
import os
import sys
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import ImageDataLoader
from src.evaluation.metrics import ModelEvaluator


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.h5 file)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)
    print("✓ Model loaded successfully\n")
    
    # Print model summary
    print("Model Summary:")
    print("-"*70)
    model.summary()
    print("-"*70 + "\n")
    
    # Load test data
    print("Loading test dataset...")
    data_loader = ImageDataLoader(
        batch_size=args.batch_size,
        validation_split=0.0,  # No split for test data
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    class_names = test_ds.class_names
    num_classes = len(class_names)
    
    print(f"✓ Test dataset loaded")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")
    print(f"  Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Evaluate model
    print("Evaluating model on test set...")
    print("-"*70)
    results = evaluator.evaluate_model(model, test_ds, verbose=1)
    print("-"*70 + "\n")
    
    print("Test Results:")
    for metric, value in results.items():
        if 'loss' in metric:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value*100:.2f}%")
    print()
    
    # Get predictions
    print("Generating predictions...")
    y_true, y_pred, y_pred_proba = evaluator.get_predictions(model, test_ds)
    print("✓ Predictions generated\n")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    model_name = Path(args.model_path).stem
    
    # Generate classification report
    print("Generating classification report...")
    report_path = os.path.join(args.results_dir, f"{model_name}_classification_report.csv")
    evaluator.generate_classification_report(
        y_true,
        y_pred,
        save_path=report_path
    )
    
    # Calculate per-class accuracy
    evaluator.calculate_per_class_accuracy(y_true, y_pred)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    cm_path = os.path.join(args.results_dir, f"{model_name}_confusion_matrix.png")
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    roc_path = os.path.join(args.results_dir, f"{model_name}_roc_curve.png")
    evaluator.plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {args.results_dir}")
    print("✓ Evaluation pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
