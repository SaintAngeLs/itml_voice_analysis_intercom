import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from train import load_data  # Reuse the data loading function from the training script

def calculate_far_frr_multi_class(y_true, y_pred, num_classes):
    """Calculate False Acceptance Ratio (FAR) and False Rejection Ratio (FRR) for multi-class."""
    far_list = []
    frr_list = []
    
    # Loop through each class and calculate FAR and FRR
    for class_index in range(num_classes):
        # Convert true labels and predictions to binary for this class (1 = class, 0 = not class)
        y_true_binary = (y_true == class_index).astype(int)
        y_pred_binary = (y_pred == class_index).astype(int)
        
        # Calculate confusion matrix for this class
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Calculate FAR and FRR
        far = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Acceptance Ratio
        frr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Rejection Ratio

        far_list.append(far)
        frr_list.append(frr)

    return far_list, frr_list

def main():
    # Load the trained model
    model = load_model('./models/cnn_model.h5')

    # Load the test data
    test_spectrogram_dir = './data/spectrograms'  # Path to the test data spectrograms
    X_test, y_test, label_encoder = load_data(test_spectrogram_dir)  # Adjusted to receive three values

    # Check if X_test is empty
    if X_test.size == 0:
        print("No test data found. Please ensure that there are spectrograms in the test directory.")
        return

    # Reshape X to match model input shape
    X_test = X_test.reshape(-1, 128, 128, 1)

    # Predict using the trained model
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)  # Use argmax to convert probabilities to class predictions

    # Calculate FAR and FRR for multi-class classification
    num_classes = len(label_encoder.classes_)
    far_list, frr_list = calculate_far_frr_multi_class(y_test, y_pred, num_classes)

    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Class: {class_name}")
        print(f"False Acceptance Ratio (FAR): {far_list[i]}")
        print(f"False Rejection Ratio (FRR): {frr_list[i]}")
        print("------")

    # Show which user was identified based on predictions
    predicted_users = label_encoder.inverse_transform(y_pred)
    actual_users = label_encoder.inverse_transform(y_test)
    for actual, predicted in zip(actual_users, predicted_users):
        print(f"Actual user: {actual}, Predicted user: {predicted}")

if __name__ == "__main__":
    main()
