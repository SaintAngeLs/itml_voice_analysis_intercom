import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class SpeakerClassifier:
    """Classifies speakers as allowed or disallowed."""
    def __init__(self, allowed_speakers):
        self.allowed_speakers = set(allowed_speakers)

    def classify(self, file_name):
        speaker_id = file_name.split('_')[0].upper()
        return 'allowed' if speaker_id in self.allowed_speakers else 'disallowed'


class AudioProcessor:
    """Handles audio loading, preprocessing, and augmentation."""
    def __init__(self, sample_rate=22050, max_duration=120, top_db=30):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.top_db = top_db

    def load_audio(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return [], self.sample_rate

        non_silent_intervals = librosa.effects.split(y, top_db=self.top_db)
        if not non_silent_intervals:
            print(f"No non-silent intervals found in {audio_path}. Skipping.")
            return [], self.sample_rate

        y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        return self._split_chunks(y_nonsilent), self.sample_rate

    def augment_audio(self, y):
        if np.random.rand() > 0.5:
            gain = np.random.uniform(0.7, 1.3)
            y = y * gain
        return y

    def _split_chunks(self, audio_data):
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio_data) > max_samples:
            return [audio_data[i:i + max_samples] for i in range(0, len(audio_data), max_samples)]
        return [audio_data]


class SpectrogramGenerator:
    """Generates and saves spectrograms."""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate(self, audio_data, sr, file_name, segment_idx):
        try:
            S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
        except Exception as e:
            print(f"Error generating spectrogram for {file_name} segment {segment_idx}: {e}")
            return None

        output_path = self._get_output_path(file_name, segment_idx)
        try:
            self._save_spectrogram(log_S, sr, output_path)
        except Exception as e:
            print(f"Error saving spectrogram for {file_name} segment {segment_idx}: {e}")
            return None
        return output_path

    def _get_output_path(self, file_name, segment_idx):
        file_dir = os.path.join(self.output_dir, f"{file_name}_segment_{segment_idx}")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        return f"{file_dir}_spectrogram.png"

    def _save_spectrogram(self, log_S, sr, output_path):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class DirectoryProcessor:
    """Processes directories and manages the overall pipeline."""
    def __init__(self, classifier, audio_processor, spectrogram_generator):
        self.classifier = classifier
        self.audio_processor = audio_processor
        self.spectrogram_generator = spectrogram_generator

    def process(self, env_path, output_dir, augment=False):
        for file_name in os.listdir(env_path):
            if not file_name.endswith('.wav') or file_name.startswith('._'):
                continue

            audio_path = os.path.join(env_path, file_name)
            class_name = self.classifier.classify(file_name)
            class_output_dir = os.path.join(output_dir, class_name)

            audio_chunks, sr = self.audio_processor.load_audio(audio_path)
            if augment:
                audio_chunks = [self.audio_processor.augment_audio(chunk) for chunk in audio_chunks]

            for idx, chunk in enumerate(audio_chunks):
                self.spectrogram_generator.generate(chunk, sr, file_name, idx)


class DatasetProcessor:
    """Coordinates dataset-wide processing."""
    def __init__(self, directory_processor):
        self.directory_processor = directory_processor

    def process(self, daps_dir, output_dir, train_dirs, test_dirs, max_duration, augment=False):
        tasks = []
        for dir_name in os.listdir(daps_dir):
            env_path = os.path.join(daps_dir, dir_name)
            if not os.path.isdir(env_path):
                continue

            if dir_name in train_dirs:
                set_dir = os.path.join(output_dir, 'train')
            elif dir_name in test_dirs:
                set_dir = os.path.join(output_dir, 'test')
            else:
                continue

            tasks.append((env_path, set_dir, augment))

        self._process_tasks(tasks)

    def _process_tasks(self, tasks):
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.directory_processor.process, *task) for task in tasks]
            with tqdm(total=len(futures), desc="Processing directories", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)


# Main function
def main():
    allowed_speakers = {'F1', 'F7', 'F8', 'M3', 'M6', 'M8'}
    train_dirs = [
        'clean', 'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 'ipad_confroom1',
        'ipad_confroom2', 'ipadflat_confroom1', 'ipadflat_office1', 'ipad_livingroom1',
        'ipad_office1', 'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 'produced'
    ]
    test_dirs = ['iphone_livingroom1']

    classifier = SpeakerClassifier(allowed_speakers)
    audio_processor = AudioProcessor()
    spectrogram_generator = SpectrogramGenerator('./data/spectrograms')
    directory_processor = DirectoryProcessor(classifier, audio_processor, spectrogram_generator)
    dataset_processor = DatasetProcessor(directory_processor)

    dataset_processor.process(
        './data/daps_data/daps', './data/spectrograms', train_dirs, test_dirs, max_duration=120, augment=True
    )


if __name__ == "__main__":
    main()
