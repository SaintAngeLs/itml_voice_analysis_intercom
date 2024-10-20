import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from train import load_data  # Reuse the data loading function from the training script

def calculate_far_frr_binary(y_true, y_pred):
    """
    Calculate False Acceptance Ratio (FAR) and False Rejection Ratio (FRR) for binary classification.
    FAR is the proportion of negative instances incorrectly classified as positive.
    FRR is the proportion of positive instances incorrectly classified as negative.
    """
    # Ensure that y_true and y_pred are integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate FAR and FRR
    far = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Acceptance Ratio
    frr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Rejection Ratio

    return far, frr

def main():
    # Load the trained model
    model = load_model('./models/cnn_model.keras')

    # Load the test data
    test_spectrogram_dir = './data/spectrograms/test'  # Test data directory
    X_test, y_test = load_data(test_spectrogram_dir)

    # Check if X_test is empty
    if X_test.size == 0:
        print("No test data found. Please ensure that there are spectrograms in the test directory.")
        return

    # Predict using the trained model
    y_pred_probs = model.predict(X_test)
    # Convert probabilities to class labels using threshold 0.5
    y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    # Calculate FAR and FRR
    far, frr = calculate_far_frr_binary(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"False Acceptance Ratio (FAR): {far * 100:.2f}%")
    print(f"False Rejection Ratio (FRR): {frr * 100:.2f}%")

    # Print confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test.astype(int), y_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

if __name__ == "__main__":
    main()
