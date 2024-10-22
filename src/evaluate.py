import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from train import load_data  # Reuse the data loading function from the training script

def calculate_far_frr_binary(y_true, y_pred):
    """
    Calculate False Acceptance Ratio (FAR) and False Rejection Ratio (FRR) for binary classification.
    FAR is the proportion of negative instances incorrectly classified as positive.
    FRR is the proportion of positive instances incorrectly classified as negative.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Acceptance Ratio
    frr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Rejection Ratio

    return far, frr, tn, fp, fn, tp

def scenario_test(model, X_test, y_test, num_allowed=100, num_disallowed=100):
    """
    Perform a real-scenario test by selecting random allowed and disallowed voices
    and testing the model's prediction on them.
    """
    # Indices of allowed and disallowed voices
    allowed_indices = np.where(y_test == 1)[0]
    disallowed_indices = np.where(y_test == 0)[0]

    # Randomly sample allowed and disallowed voices
    allowed_samples = random.sample(list(allowed_indices), min(num_allowed, len(allowed_indices)))
    disallowed_samples = random.sample(list(disallowed_indices), min(num_disallowed, len(disallowed_indices)))

    # Combine selected samples and their true labels
    selected_indices = np.concatenate([allowed_samples, disallowed_samples])
    X_selected = X_test[selected_indices]
    y_selected = y_test[selected_indices]

    # Predict using the trained model
    y_pred_probs = model.predict(X_selected)
    y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

    # Output the results
    for i, index in enumerate(selected_indices):
        actual_class = 'allowed' if y_selected[i] == 1 else 'disallowed'
        predicted_class = 'allowed' if y_pred[i] == 1 else 'disallowed'
        print(f"Voice {index}: Actual - {actual_class}, Predicted - {predicted_class}")

    # Calculate accuracy for the scenario test
    scenario_accuracy = np.mean(y_pred == y_selected)
    print(f"Scenario Test Accuracy: {scenario_accuracy * 100:.2f}%")

def main():
    # Load the trained model
    model = load_model('./final_model.keras')

    # Load the test data
    test_spectrogram_dir = './data/spectrograms/test'  # Test data directory
    X_test, y_test = load_data(test_spectrogram_dir)

    # Check if X_test is empty
    if X_test.size == 0:
        print("No test data found. Please ensure that there are spectrograms in the test directory.")
        return

    # Evaluate the model using the evaluate method
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict using the trained model
    y_pred_probs = model.predict(X_test)
    
    # Convert probabilities to class labels using threshold 0.5
    threshold = 0.5
    y_pred = (y_pred_probs >= threshold).astype(int).reshape(-1)

    # Calculate FAR, FRR, and confusion matrix components
    far, frr, tn, fp, fn, tp = calculate_far_frr_binary(y_test, y_pred)

    print(f"False Acceptance Ratio (FAR): {far * 100:.2f}%")
    print(f"False Rejection Ratio (FRR): {frr * 100:.2f}%")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    # Perform scenario test with random samples
    print("\nPerforming scenario test with random samples...")
    scenario_test(model, X_test, y_test, num_allowed=20, num_disallowed=20)

if __name__ == "__main__":
    main()
