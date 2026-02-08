"""
Evaluation metrics and visualization utilities.

Provides comprehensive model evaluation including confusion matrix,
ROC curves, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score
)
import pandas as pd


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit.
    
    Provides methods for:
    - Confusion matrix visualization
    - ROC curve plotting
    - Classification reports
    - Performance metrics calculation
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate_model(
        self,
        model: keras.Model,
        test_dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> dict:
        """
        Evaluate model on test dataset.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate
        results = model.evaluate(test_dataset, verbose=verbose)
        
        # Get metric names
        metric_names = model.metrics_names
        
        # Create results dictionary
        results_dict = {
            name: value
            for name, value in zip(metric_names, results)
        }
        
        return results_dict
    
    def get_predictions(
        self,
        model: keras.Model,
        dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for dataset.
        
        Args:
            model: Trained model
            dataset: Dataset to predict on
            
        Returns:
            Tuple of (y_true, y_pred, y_pred_proba)
        """
        # Get true labels and predictions
        y_true = []
        y_pred_proba = []
        
        for images, labels in dataset:
            predictions = model.predict(images, verbose=0)
            y_pred_proba.extend(predictions)
            
            # Convert one-hot to class indices if needed
            if len(labels.shape) > 1:
                y_true.extend(np.argmax(labels.numpy(), axis=1))
            else:
                y_true.extend(labels.numpy())
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            figsize: Figure size
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot ROC curves for all classes.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save figure
            figsize: Figure size
        """
        # Convert to one-hot if needed
        from sklearn.preprocessing import label_binarize
        y_true_binary = label_binarize(y_true, classes=range(self.num_classes))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr,
                label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report as CSV
            
        Returns:
            DataFrame with classification metrics
        """
        # Generate report
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(report_dict).transpose()
        
        # Display
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(df.to_string())
        print("="*70 + "\n")
        
        if save_path:
            df.to_csv(save_path)
            print(f"Classification report saved to: {save_path}")
        
        return df
    
    def plot_training_history(
        self,
        history: keras.callbacks.History,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Training history object
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to: {save_path}")
        
        plt.show()
    
    def calculate_per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate per-class accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        results = []
        
        for i, class_name in enumerate(self.class_names):
            mask = y_true == i
            class_acc = np.mean(y_pred[mask] == y_true[mask]) * 100
            class_count = np.sum(mask)
            
            results.append({
                'Class': class_name,
                'Accuracy (%)': f"{class_acc:.2f}",
                'Sample Count': class_count
            })
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*50)
        print("PER-CLASS ACCURACY")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50 + "\n")
        
        return df


def compare_models(
    results: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare multiple models performance.
    
    Args:
        results: Dictionary with model_name: metrics
        save_path: Path to save comparison plot
        figsize: Figure size
    """
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=figsize)
    
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()
    
    # Print table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(df.to_string())
    print("="*70 + "\n")
