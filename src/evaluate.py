"""
Audio Evaluation Script for Speaker Classification

This script evaluates a trained machine learning model for speaker classification using audio data. 
It performs tasks such as loading audio files, processing them into mel-spectrograms, predicting classes, 
and calculating evaluation metrics. The script also includes functionality for scenario testing with specific 
audio samples, allowing for thorough assessment of the model's performance on various test cases.

Key Components:

- Imports: Essential libraries for audio processing (librosa), numerical operations (numpy), 
  machine learning model loading (tensorflow), and visualization (matplotlib). It also includes 
  tools for calculating confusion matrices and evaluation metrics.

- Allowed Speakers:
  A predefined set of allowed speakers (class_1_speakers) is used to classify audio files into 
  "allowed" or "disallowed" categories based on the speaker ID extracted from the filename.

- Evaluation Functions:
  - `calculate_far_frr_binary`: Computes the False Acceptance Ratio (FAR) and False Rejection Ratio (FRR)
    for binary classification, along with confusion matrix components (TP, TN, FP, FN).
  - `calculate_general_efficiency_coefficient`: Calculates the General Efficiency Coefficient (GEC) 
    based on accuracy, FAR, and FRR, using specified weights.

- Audio Processing:
  - `load_audio_chunk`: Loads an audio file, removes silent sections, and extracts a specified 
    duration of audio.
  - `generate_mel_spectrogram`: Generates a mel-spectrogram from the audio data, ensuring it fits a 
    defined target shape for input into the model.

- Scenario Testing:
  - `load_random_audio_data_from_livingroom`: Loads random audio samples from a specified directory, 
    converts them to mel-spectrograms, and ensures consistent shape for the model's input.
  - `scenario_test`: Tests the model on randomly selected allowed and disallowed samples, outputting 
    predictions and accuracy information.
  - `scenario_test_on_audio_with_duration`: Conducts scenario tests using audio files of a specific 
    duration, predicting classes and evaluating results.

- Main Functionality:
  - The `main` function orchestrates the evaluation process, loading the trained model, preparing the test 
    data, evaluating model performance, and printing detailed results.
"""

import os
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train import load_data  # Reuse the data loading function from the training script
import random

# Allowed speakers
class_1_speakers = {'F1', 'F7', 'F8', 'M3', 'M6', 'M8'}

def calculate_far_frr_binary(y_true, y_pred):
    """
    Calculate False Acceptance Ratio (FAR) and False Rejection Ratio (FRR) for binary classification.
    
    Parameters:
    - y_true: Ground truth labels (1 for allowed, 0 for disallowed)
    - y_pred: Model predictions (1 for allowed, 0 for disallowed)
    
    Returns:
    - FAR: False Acceptance Ratio (disallowed incorrectly classified as allowed)
    - FRR: False Rejection Ratio (allowed incorrectly classified as disallowed)
    - tn: True negatives count
    - fp: False positives count
    - fn: False negatives count
    - tp: True positives count
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Acceptance Ratio
    frr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Rejection Ratio

    return far, frr, tn, fp, fn, tp

def calculate_general_efficiency_coefficient(accuracy, far, frr, w1=0.4, w2=0.3, w3=0.3):
    """
    Calculate the General Efficiency Coefficient (GEC) as a weighted average of accuracy, (1 - FAR), and (1 - FRR).
    
    Parameters:
    - accuracy: Overall accuracy of the model
    - far: False Acceptance Ratio
    - frr: False Rejection Ratio
    - w1, w2, w3: Weights for accuracy, (1 - FAR), and (1 - FRR), respectively
    
    Returns:
    - gec: Calculated General Efficiency Coefficient
    """
    gec = w1 * accuracy + w2 * (1 - far) + w3 * (1 - frr)
    return gec

def load_audio_chunk(audio_path, sr=22050, duration=10, top_db=30):
    """
    Load an audio file, remove silent sections, and extract a specific duration in seconds.
    
    Parameters:
    - audio_path: Path to the audio file
    - sr: Sample rate to load the audio at
    - duration: Duration of audio to extract in seconds
    - top_db: Threshold in decibels for determining silence
    
    Returns:
    - y_nonsilent: Non-silent audio chunk within the specified duration
    - sr: Sample rate of the loaded audio
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None

    # Remove silent parts of the audio
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    if len(non_silent_intervals) == 0:
        print(f"No non-silent intervals found in {audio_path}. Skipping.")
        return None, None

    # Concatenate non-silent intervals
    y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    # Ensure that we return only the first `duration` seconds
    max_samples = int(duration * sr)
    y_nonsilent = y_nonsilent[:max_samples]
    
    return y_nonsilent, sr

def generate_mel_spectrogram(audio_data, sr, output_shape=(128, 128), n_mels=128, fmax=8000):
    """
    Generate a mel-spectrogram from audio data and ensure it fits the target shape.
    
    Parameters:
    - audio_data: Audio data array
    - sr: Sample rate of the audio data
    - output_shape: Desired shape for the output spectrogram (height, width)
    - n_mels: Number of mel bands
    - fmax: Maximum frequency for the mel spectrogram
    
    Returns:
    - log_mel_spectrogram: Logarithmic mel-spectrogram with the target shape
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Pad or truncate the spectrogram to the required shape
    if log_mel_spectrogram.shape[1] < output_shape[1]:
        # Pad with zeros
        padding = output_shape[1] - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
    else:
        # Truncate to the correct number of frames
        log_mel_spectrogram = log_mel_spectrogram[:, :output_shape[1]]

    return log_mel_spectrogram

def load_random_audio_data_from_livingroom(num_samples=20, livingroom_dir='./data/daps/iphone_livingroom1', output_shape=(128, 128)):
    """
    Load random audio samples from a directory, convert to mel-spectrograms, and ensure consistent shape.
    
    Parameters:
    - num_samples: Number of samples to load
    - livingroom_dir: Directory containing audio files
    - output_shape: Desired shape for each mel-spectrogram
    
    Returns:
    - X_audio: Array of mel-spectrograms with the specified shape
    - selected_files: List of filenames corresponding to the loaded samples
    """
    if not os.path.exists(livingroom_dir):
        print(f"Error: Directory {livingroom_dir} does not exist.")
        return [], []

    audio_files = [f for f in os.listdir(livingroom_dir) if f.endswith('.wav')]
    
    if len(audio_files) == 0:
        print(f"No audio files found in {livingroom_dir}")
        return [], []
        
    selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    X_audio = []
    for file in selected_files:
        file_path = os.path.join(livingroom_dir, file)
        
        # Load the audio file and extract the specific duration
        audio_data, sr = load_audio_chunk(file_path, duration=10)
        if audio_data is None:
            continue
        
        # Generate mel-spectrogram and ensure correct frame size (128x128)
        mel_spectrogram_padded = generate_mel_spectrogram(audio_data, sr, output_shape=output_shape)
        
        # Expand dimensions for CNN input shape compatibility
        mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=-1)  # Add channel dimension
        
        X_audio.append(mel_spectrogram_padded)

    return np.array(X_audio), selected_files

def extract_speaker_id(filename):
    """
    Extract the speaker ID from a filename (e.g., 'M3', 'F1').
    
    Parameters:
    - filename: Name of the audio file (e.g., 'm3_script5_iphone_livingroom1.wav')
    
    Returns:
    - Speaker ID in uppercase format
    """
    return filename.split('_')[0].upper()  # Extract 'M3', 'F1', etc.

def scenario_test_on_audio_with_duration(model, livingroom_dir, num_samples=20, duration=10):
    """
    Perform a scenario test by selecting audio files of a specific duration, predicting, and evaluating.
    
    Parameters:
    - model: Trained Keras model
    - livingroom_dir: Directory containing audio files for testing
    - num_samples: Number of audio files to test
    - duration: Duration in seconds for each audio sample
    
    Returns:
    - None, prints the test results
    """
    # Load random audio samples of the specified duration from the living room
    X_audio, selected_files = load_random_audio_data_from_livingroom(num_samples=num_samples, livingroom_dir=livingroom_dir)
    
    if len(X_audio) == 0:
        print("No audio data available for testing.")
        return

    # Predict using the trained model
    y_pred_probs = model.predict(X_audio)
    y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

    # Track the number of correct and incorrect predictions
    correct_predictions = 0
    incorrect_predictions = 0

    # Output the results with correct/incorrect prediction information
    for i, file in enumerate(selected_files):
        predicted_class = 'allowed' if y_pred[i] == 1 else 'disallowed'
        speaker_id = extract_speaker_id(file)
        actual_class = 'allowed' if speaker_id in class_1_speakers else 'disallowed'
        
        correctness = 'correct' if predicted_class == actual_class else 'incorrect'
        
        if correctness == 'correct':
            correct_predictions += 1
        else:
            incorrect_predictions += 1
        
        print(f"Audio file {file}: Predicted - {predicted_class}, Actual - {actual_class}, Prediction is {correctness}")

    # Calculate percentages
    total_predictions = correct_predictions + incorrect_predictions
    if total_predictions > 0:
        correct_percentage = (correct_predictions / total_predictions) * 100
        incorrect_percentage = (incorrect_predictions / total_predictions) * 100

        print(f"\nSummary:")
        print(f"Correct Predictions: {correct_predictions}/{total_predictions} ({correct_percentage:.2f}%)")
        print(f"Incorrect Predictions: {incorrect_predictions}/{total_predictions} ({incorrect_percentage:.2f}%)")
    else:
        print("No valid predictions were made.")

def scenario_test(model, X_test, y_test, num_allowed=20, num_disallowed=20):
    """
    Perform a test on randomly selected allowed and disallowed samples, outputting predictions.
    
    Parameters:
    - model: Trained Keras model
    - X_test: Feature data for the test set
    - y_test: True labels for the test set
    - num_allowed: Number of allowed samples to test
    - num_disallowed: Number of disallowed samples to test
    
    Returns:
    - None, prints predictions and accuracy information
    """
    allowed_indices = np.where(y_test == 1)[0]
    disallowed_indices = np.where(y_test == 0)[0]

    allowed_samples = random.sample(list(allowed_indices), min(num_allowed, len(allowed_indices)))
    disallowed_samples = random.sample(list(disallowed_indices), min(num_disallowed, len(disallowed_indices)))

    selected_indices = np.concatenate([allowed_samples, disallowed_samples])
    X_selected = X_test[selected_indices]
    y_selected = y_test[selected_indices]

    y_pred_probs = model.predict(X_selected)
    y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

    for i, index in enumerate(selected_indices):
        actual_class = 'allowed' if y_selected[i] == 1 else 'disallowed'
        predicted_class = 'allowed' if y_pred[i] == 1 else 'disallowed'
        print(f"Voice {index}: Actual - {actual_class}, Predicted - {predicted_class}")

    scenario_accuracy = np.mean(y_pred == y_selected)
    print(f"Scenario Test Accuracy: {scenario_accuracy * 100:.2f}%")

# Main function to call the evaluation functions
def main():
    # Load the trained model
    model = load_model('./final_model_2.keras')

    # Load the test data
    test_spectrogram_dir = './data/spectrograms/test'
    X_test, y_test = load_data(test_spectrogram_dir)

    if X_test.size == 0:
        print("No test data found. Please ensure that there are spectrograms in the test directory.")
        return

    # Variables to accumulate results for GEC calculation
    overall_accuracy = []
    overall_far = []
    overall_frr = []

    # Evaluate the model using the evaluate method
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict using the trained model
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).reshape(-1)

    # Calculate FAR, FRR, and confusion matrix components
    far, frr, tn, fp, fn, tp = calculate_far_frr_binary(y_test, y_pred)
    print(f"False Acceptance Ratio (FAR): {far * 100:.2f}%")
    print(f"False Rejection Ratio (FRR): {frr * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    # Append to overall results for final GEC calculation
    overall_accuracy.append(test_accuracy)
    overall_far.append(far)
    overall_frr.append(frr)

    # Perform scenario test with random samples from test data
    print("\nPerforming scenario test with random samples from test data...")
    scenario_test(model, X_test, y_test, num_allowed=20, num_disallowed=20)

    # Perform scenario test on audio with specific duration
    print("\nPerforming scenario test with audio chunks of 20 seconds from living room...")
    scenario_test_on_audio_with_duration(model, './data/daps_data/daps/iphone_livingroom1', num_samples=100, duration=20)

    # Add scenario test results to GEC calculation
    # (You would gather FAR, FRR, Accuracy from these tests similarly and append)

    print("\nPerforming scenario test with audio chunks of 10 seconds from living room...")
    scenario_test_on_audio_with_duration(model, './data/daps_data/daps/iphone_livingroom1', num_samples=100, duration=10)

    print("\nPerforming scenario test with audio chunks of 5 seconds from living room...")
    scenario_test_on_audio_with_duration(model, './data/daps_data/daps/iphone_livingroom1', num_samples=100, duration=5)

    # Final GEC Calculation after all tests
    avg_accuracy = np.mean(overall_accuracy)
    avg_far = np.mean(overall_far)
    avg_frr = np.mean(overall_frr)

    gec = calculate_general_efficiency_coefficient(avg_accuracy, avg_far, avg_frr)
    print(f"\nFinal General Efficiency Coefficient (GEC): {gec:.2f}")

if __name__ == "__main__":
    main()
