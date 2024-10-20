import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import noisereduce as nr
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define Class 1 (allowed) and Class 0 (disallowed) speakers
class_1_speakers = {'F1', 'F7', 'F8', 'M3', 'M6', 'M8'}  # Allowed users for training
class_0_speakers = set()  # Dynamically populated disallowed speakers

# Define the directories for training and testing
train_dirs = [
    'clean', 'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 'ipad_confroom1',
    'ipad_confroom2', 'ipadflat_confroom1', 'ipadflat_office1', 'ipad_livingroom1',
    'ipad_office1', 'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 'produced'
]
test_dirs = ['iphone_livingroom1']


def load_audio(audio_path, sr=22050, max_duration=10, top_db=30, augment=False):
    """Load audio file, remove silence, and return segments if the audio is longer than max_duration seconds."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return [], sr

    # Remove silent parts of the audio
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    if len(non_silent_intervals) == 0:
        print(f"No non-silent intervals found in {audio_path}. Skipping.")
        return [], sr
    y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    # Optionally augment the audio data
    if augment:
        y_nonsilent = augment_audio(y_nonsilent, sr)

    max_samples = int(max_duration * sr)  # Maximum samples for the duration

    # If the audio is longer than max_duration, split it into chunks
    if len(y_nonsilent) > max_samples:
        audio_chunks = [y_nonsilent[i:i + max_samples] for i in range(0, len(y_nonsilent), max_samples)]
    else:
        audio_chunks = [y_nonsilent]

    return audio_chunks, sr

def augment_audio(y, sr):
    """Apply random augmentations to the audio data."""
    if random.random() > 0.5:
        gain = random.uniform(0.7, 1.3)
        y = y * gain
    if random.random() > 0.5:
        rate = random.uniform(0.8, 1.2)
        try:
            y = librosa.effects.time_stretch(y, rate=rate)
        except Exception as e:
            print(f"Error during time stretching: {e}")
    if random.random() > 0.5:
        n_steps = random.randint(-4, 4)
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        except Exception as e:
            print(f"Error during pitch shifting: {e}")
    if random.random() > 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
    if random.random() > 0.5:
        y = apply_time_masking(y, sr)
    if random.random() > 0.5:
        y = apply_frequency_masking(y, sr)
    if random.random() > 0.5:
        y = add_reverb(y, sr)
    return y

def apply_time_masking(y, sr, max_mask_pct=0.1):
    """Apply random time masking."""
    max_mask_len = int(max_mask_pct * len(y))
    if max_mask_len == 0:
        return y
    mask_start = random.randint(0, max(len(y) - max_mask_len, 1))
    y[mask_start:mask_start + max_mask_len] = 0
    return y

def apply_frequency_masking(y, sr, max_mask_pct=0.1):
    """Apply random frequency masking."""
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    num_freq_bins = S_db.shape[0]
    mask_size = int(num_freq_bins * max_mask_pct)
    if mask_size == 0:
        return y
    mask_start = random.randint(0, max(num_freq_bins - mask_size, 1))
    S_db[mask_start:mask_start + mask_size, :] = 0
    y_masked = librosa.istft(librosa.db_to_amplitude(S_db))
    return y_masked

def add_reverb(y, sr, reverb_factor=0.2):
    """Add reverberation."""
    reverb = np.convolve(y, np.random.randn(int(reverb_factor * sr)), mode='full')
    reverb = reverb[:len(y)]
    y_reverb = y + 0.5 * reverb  # Mix original and reverberated signal
    return y_reverb

def generate_spectrogram(audio_data, sr, output_dir, file_name, segment_idx):
    """Generate a mel-spectrogram and save it."""
    try:
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        print(f"Error generating spectrogram for {file_name} segment {segment_idx}: {e}")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{file_name}_segment_{segment_idx}_spectrogram.png")
    try:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    except Exception as e:
        print(f"Error saving spectrogram for {file_name} segment {segment_idx}: {e}")
        return None
    return output_file

def classify_speaker(file_name):
    """Classify speaker as allowed (Class 1) or disallowed (Class 0) based on the file name."""
    speaker_id = file_name.split('_')[0].upper()
    if speaker_id in class_1_speakers:
        return 'allowed'
    else:
        class_0_speakers.add(speaker_id)  # Dynamically populate class_0_speakers
        return 'disallowed'

def process_directory(env_path, output_dir, max_duration=10, augment=False):
    """Process a directory of files for training or testing."""
    for file_name in os.listdir(env_path):
        if file_name.startswith('._') or not file_name.endswith('.wav'):
            continue  # Skip hidden files and non-wav files

        audio_path = os.path.join(env_path, file_name)
        class_name = classify_speaker(file_name)
        class_output_dir = os.path.join(output_dir, class_name)

        # Load and optionally augment the audio
        audio_chunks, sr = load_audio(audio_path, max_duration=max_duration, augment=augment)

        # Save spectrograms for each chunk
        for idx, chunk in enumerate(audio_chunks):
            spectrogram_file = generate_spectrogram(chunk, sr, class_output_dir, file_name, idx)
            if spectrogram_file:
                print(f"Spectrogram saved to {spectrogram_file}")

def process_daps_data(daps_dir, output_dir, max_duration=10, augment=False):
    """Process all audio files from the DAPS dataset, classify speakers, and split into train/test sets based on directories."""
    
    # Prepare lists of directories to process for training and testing
    files_to_process = []

    for env_dir in os.listdir(daps_dir):
        env_path = os.path.join(daps_dir, env_dir)
        if not os.path.isdir(env_path):
            continue

        # Determine whether the directory is for training or testing
        if env_dir in train_dirs:
            output_set_dir = os.path.join(output_dir, 'train')
            augment_flag = augment  # Apply augmentation to training data
        elif env_dir in test_dirs:
            output_set_dir = os.path.join(output_dir, 'test')
            augment_flag = False  # Do not augment test data
        else:
            continue  # Skip if not in train or test directories

        files_to_process.append((env_path, output_set_dir, max_duration, augment_flag))
    
    print(f"Total directories to process: {len(files_to_process)}")

    # Use ProcessPoolExecutor with tqdm for progress bar
    print("Starting data processing...")
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_directory, *args) for args in files_to_process]
        
        # Wrap the futures in tqdm for progress bar
        with tqdm(total=len(futures), desc="Processing files", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred during processing: {e}")
                finally:
                    pbar.update(1)

def main():
    # Paths to the DAPS dataset and output folder
    daps_dir = './data/daps_data/daps'  # Path to DAPS data
    output_dir = './data/spectrograms'
    
    # Process DAPS data with split into train and test
    process_daps_data(daps_dir, output_dir, max_duration=10, augment=True)

if __name__ == "__main__":
    main()
