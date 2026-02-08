#!/usr/bin/env python3
"""
Prediction script for trained models.

Usage:
    python scripts/predict.py --model_path models/saved_models/resnet50_best.h5 --image_path path/to/image.jpg
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Load and preprocess image for prediction.
    
    Args:
        image_path: Path to image file
        target_size: Target size for image
        
    Returns:
        Preprocessed image array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img


def predict_image(
    model: tf.keras.Model,
    image_path: str,
    class_names: list = None,
    top_k: int = 5
):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        class_names: List of class names
        top_k: Number of top predictions to show
        
    Returns:
        Tuple of (predictions, original_image)
    """
    # Load and preprocess image
    img_array, original_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    
    # Print results
    print("\n" + "="*70)
    print(f"PREDICTIONS FOR: {os.path.basename(image_path)}")
    print("="*70)
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"{i}. {class_name}: {prob*100:.2f}%")
    
    print("="*70 + "\n")
    
    return predictions, original_img, top_indices, top_probs


def visualize_prediction(
    image,
    top_indices,
    top_probs,
    class_names,
    save_path=None
):
    """
    Visualize image with predictions.
    
    Args:
        image: Original image
        top_indices: Indices of top predictions
        top_probs: Probabilities of top predictions
        class_names: List of class names
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Show predictions
    class_labels = [
        class_names[idx] if class_names else f"Class {idx}"
        for idx in top_indices
    ]
    
    colors = plt.cm.Blues(top_probs / top_probs.max())
    y_pos = np.arange(len(top_indices))
    
    axes[1].barh(y_pos, top_probs * 100, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_labels)
    axes[1].set_xlabel('Confidence (%)', fontsize=12)
    axes[1].set_title('Top Predictions', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    # Add percentage labels
    for i, (prob, color) in enumerate(zip(top_probs, colors)):
        axes[1].text(
            prob * 100 + 1,
            i,
            f'{prob*100:.1f}%',
            va='center',
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def batch_predict(
    model: tf.keras.Model,
    image_dir: str,
    class_names: list = None,
    top_k: int = 3,
    save_results: bool = True,
    results_dir: str = "results/predictions"
):
    """
    Make predictions on all images in a directory.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        class_names: List of class names
        top_k: Number of top predictions to show
        save_results: Whether to save results
        results_dir: Directory to save results
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    print(f"\nFound {len(image_files)} images in {image_dir}\n")
    
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Make prediction
        predictions, original_img, top_indices, top_probs = predict_image(
            model,
            image_path,
            class_names,
            top_k
        )
        
        # Save visualization
        if save_results:
            save_path = os.path.join(
                results_dir,
                f"{os.path.splitext(image_file)[0]}_prediction.png"
            )
            visualize_prediction(
                original_img,
                top_indices,
                top_probs,
                class_names,
                save_path
            )
        else:
            visualize_prediction(
                original_img,
                top_indices,
                top_probs,
                class_names
            )


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict with trained model')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.h5 file)'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to single image file'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory of images (for batch prediction)'
    )
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        help='List of class names (optional)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save prediction visualizations'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/predictions',
        help='Directory to save results'
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        parser.error("Must provide either --image_path or --image_dir")
    
    print("="*70)
    print("IMAGE CLASSIFICATION PREDICTION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)
    print("✓ Model loaded successfully\n")
    
    # Single image prediction
    if args.image_path:
        predictions, original_img, top_indices, top_probs = predict_image(
            model,
            args.image_path,
            args.class_names,
            args.top_k
        )
        
        # Visualize
        save_path = None
        if args.save_results:
            os.makedirs(args.results_dir, exist_ok=True)
            save_path = os.path.join(
                args.results_dir,
                f"{Path(args.image_path).stem}_prediction.png"
            )
        
        visualize_prediction(
            original_img,
            top_indices,
            top_probs,
            args.class_names,
            save_path
        )
    
    # Batch prediction
    elif args.image_dir:
        batch_predict(
            model,
            args.image_dir,
            args.class_names,
            args.top_k,
            args.save_results,
            args.results_dir
        )
    
    print("\n✓ Prediction completed successfully!\n")


if __name__ == "__main__":
    main()
