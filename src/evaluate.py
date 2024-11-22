import os
import numpy as np
import librosa
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from train import DataLoader
import random


class AudioProcessor:
    """Processes audio files for evaluation."""

    def __init__(self, sr=22050, duration=10, top_db=30):
        """
        Initialize the AudioProcessor.

        Parameters:
        - sr: Sample rate to load the audio at.
        - duration: Duration of audio to extract in seconds.
        - top_db: Threshold in decibels for determining silence.
        """
        self.sr = sr
        self.duration = duration
        self.top_db = top_db

    def load_audio_chunk(self, audio_path):
        """
        Load an audio file, remove silent sections, and extract a specific duration in seconds.

        Parameters:
        - audio_path: Path to the audio file.

        Returns:
        - y_nonsilent: Non-silent audio chunk within the specified duration.
        - sr: Sample rate of the loaded audio.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None

        # Remove silent parts of the audio
        non_silent_intervals = librosa.effects.split(y, top_db=self.top_db)
        if not non_silent_intervals:
            print(f"No non-silent intervals found in {audio_path}. Skipping.")
            return None, None

        # Concatenate non-silent intervals
        y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])

        # Ensure that we return only the first `duration` seconds
        max_samples = int(self.duration * sr)
        y_nonsilent = y_nonsilent[:max_samples]

        return y_nonsilent, sr

    def generate_mel_spectrogram(self, audio_data, sr, output_shape=(128, 128), n_mels=128, fmax=8000):
        """
        Generate a mel-spectrogram from audio data and ensure it fits the target shape.

        Parameters:
        - audio_data: Audio data array.
        - sr: Sample rate of the audio data.

        Returns:
        - log_mel_spectrogram: Logarithmic mel-spectrogram with the target shape.
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels, fmax=fmax)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Pad or truncate the spectrogram to the required shape
        if log_mel_spectrogram.shape[1] < output_shape[1]:
            padding = output_shape[1] - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
        else:
            log_mel_spectrogram = log_mel_spectrogram[:, :output_shape[1]]

        return log_mel_spectrogram


class EvaluationMetrics:
    """Handles evaluation metric calculations."""

    @staticmethod
    def calculate_far_frr_binary(y_true, y_pred):
        """Calculate FAR and FRR for binary classification."""
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) != 0 else 0
        frr = fn / (fn + tp) if (fn + tp) != 0 else 0

        return far, frr, tn, fp, fn, tp

    @staticmethod
    def calculate_general_efficiency_coefficient(accuracy, far, frr, w1=0.4, w2=0.3, w3=0.3):
        """Calculate the General Efficiency Coefficient (GEC)."""
        return w1 * accuracy + w2 * (1 - far) + w3 * (1 - frr)


class ScenarioTester:
    """Performs scenario testing on audio data."""

    def __init__(self, model, audio_processor, class_1_speakers):
        """
        Initialize ScenarioTester.

        Parameters:
        - model: Trained model to evaluate.
        - audio_processor: Instance of AudioProcessor.
        - class_1_speakers: Set of allowed speakers.
        """
        self.model = model
        self.audio_processor = audio_processor
        self.class_1_speakers = class_1_speakers

    def test_on_audio_with_duration(self, livingroom_dir, num_samples=20, duration=10):
        """Perform a scenario test using audio files of specific duration."""
        files = [f for f in os.listdir(livingroom_dir) if f.endswith('.wav')]
        selected_files = random.sample(files, min(num_samples, len(files)))

        correct_predictions = 0
        incorrect_predictions = 0

        for file in selected_files:
            file_path = os.path.join(livingroom_dir, file)
            audio_data, sr = self.audio_processor.load_audio_chunk(file_path)

            if audio_data is None:
                continue

            spectrogram = self.audio_processor.generate_mel_spectrogram(audio_data, sr)
            spectrogram = np.expand_dims(spectrogram, axis=[0, -1])  # Model expects batch and channel dimensions

            prediction = self.model.predict(spectrogram)
            predicted_class = 1 if prediction >= 0.5 else 0

            speaker_id = file.split('_')[0].upper()
            actual_class = 1 if speaker_id in self.class_1_speakers else 0

            if predicted_class == actual_class:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

        total = correct_predictions + incorrect_predictions
        if total > 0:
            print(f"Correct: {correct_predictions}, Incorrect: {incorrect_predictions}")
            print(f"Accuracy: {correct_predictions / total * 100:.2f}%")
        else:
            print("No valid audio samples for evaluation.")

    def test_random_samples(self, X_test, y_test, num_allowed=20, num_disallowed=20):
        """Test the model on random samples of allowed and disallowed speakers."""
        allowed_indices = np.where(y_test == 1)[0]
        disallowed_indices = np.where(y_test == 0)[0]

        allowed_samples = random.sample(list(allowed_indices), min(num_allowed, len(allowed_indices)))
        disallowed_samples = random.sample(list(disallowed_indices), min(num_disallowed, len(disallowed_indices)))

        selected_indices = np.concatenate([allowed_samples, disallowed_samples])
        X_selected = X_test[selected_indices]
        y_selected = y_test[selected_indices]

        predictions = self.model.predict(X_selected).reshape(-1)
        predictions = (predictions >= 0.5).astype(int)

        accuracy = np.mean(predictions == y_selected) * 100
        print(f"Scenario Test Accuracy: {accuracy:.2f}%")


def main():
    """Main function for evaluation."""
    model_path = './outputs/final_model.keras'
    model = load_model(model_path)

    # Load test data
    data_loader = DataLoader('./data/spectrograms/test')
    X_test, y_test = data_loader.load_data()

    audio_processor = AudioProcessor(duration=10)
    metrics_calculator = EvaluationMetrics()
    scenario_tester = ScenarioTester(model, audio_processor, class_1_speakers={'F1', 'F7', 'F8', 'M3', 'M6', 'M8'})

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    # FAR and FRR
    predictions = model.predict(X_test).reshape(-1)
    predictions = (predictions >= 0.5).astype(int)
    far, frr, tn, fp, fn, tp = metrics_calculator.calculate_far_frr_binary(y_test, predictions)

    print(f"False Acceptance Ratio (FAR): {far * 100:.2f}%")
    print(f"False Rejection Ratio (FRR): {frr * 100:.2f}%")

    # Scenario Tests
    print("\nPerforming scenario tests on random samples:")
    scenario_tester.test_random_samples(X_test, y_test)

    print("\nTesting on audio chunks from living room samples:")
    scenario_tester.test_on_audio_with_duration('./data/daps/iphone_livingroom1', num_samples=100, duration=10)


if __name__ == "__main__":
    main()
