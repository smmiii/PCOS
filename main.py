import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from pcos_detection.data_loader import load_images
from pcos_detection.model import build_and_train_model
from pcos_detection.utils import plot_results

def main():
    # Load data
    X_train, y_train = load_images('image/train')
    X_test, y_test = load_images('image/test')
    
    print(f"Shape of X_train (raw): {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Number of samples in training set: {len(X_train)}")
    
    # Ensure images have correct shape (samples, height, width, channels)
    # Assuming RGB images (3 channels)
    if X_train.ndim == 3:  # If channels are missing
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)

    print(f"Shape of X_train after reshaping: {X_train.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 64 * 64 * 3)).reshape(-1, 64, 64, 3)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 64 * 64 * 3)).reshape(-1, 64, 64, 3)

    print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
    print(f"Shape of X_test_scaled: {X_test_scaled.shape}")

    # Split train set into training and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    print(f"Shape of X_train_final: {X_train_final.shape}")
    print(f"Shape of X_val: {X_val.shape}")

    # Build and train model
    model, history = build_and_train_model(X_train_final, y_train_final, X_val, y_val)

    # Save the trained model
    model_save_path = 'saved_model/pcos_detection_model.h5'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.2f}')

    # Plot results
    plot_results(history, model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
