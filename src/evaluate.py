import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from train import load_data

def calculate_far_frr(y_true, y_pred):
    """Calculate False Acceptance Ratio (FAR) and False Rejection Ratio (FRR)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    far = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Acceptance Ratio
    frr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Rejection Ratio

    return far, frr

def main():
    # Load the trained model
    model = load_model('./models/cnn_model.h5')

    # Load the test data
    spectrogram_dir = './data/spectrograms'
    X_test, y_test = load_data(spectrogram_dir)

    # Check if X_test is empty
    if X_test.size == 0:
        print("No test data found. Please ensure that there are spectrograms in the test directory.")
        return

    # Reshape X to match model input
    X_test = X_test.reshape(-1, 128, 128, 1)

    # Predict
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate FAR and FRR
    if len(y_test) == 0 or len(y_pred) == 0:
        print("Prediction or ground truth is empty. FAR and FRR cannot be calculated.")
        return

    far, frr = calculate_far_frr(y_test, y_pred)
    print(f"False Acceptance Ratio: {far}")
    print(f"False Rejection Ratio: {frr}")

if __name__ == "__main__":
    main()
