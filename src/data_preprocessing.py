"""
Audio Preprocessing Pipeline with Spectrogram Generation

This script processes audio files to prepare them for machine learning tasks. It loads audio data, 
removes silent sections, augments the data, and generates mel-spectrograms for use in training and testing 
classification models. It supports efficient parallel processing of audio files from specified directories.

Key Components:

- Imports: Essential libraries for audio processing (librosa), numerical operations (numpy), and 
  visualization (matplotlib). The script also includes parallel processing tools for handling large 
  datasets efficiently.

- Speaker Classification:
  A predefined set of allowed speakers (class_1_speakers) is used to classify audio files into 
  "allowed" or "disallowed" categories based on the speaker ID extracted from the filename.

- Audio Loading and Preprocessing:
  - The `load_audio` function loads an audio file, removes silent sections, and splits it into chunks if 
    it exceeds a specified duration.
  - `augment_audio` function applies random gain adjustments to audio samples to increase data variability.

- Spectrogram Generation:
  - `generate_spectrogram` function generates mel-spectrograms from audio data and saves them as PNG files 
    for easy visualization and further processing.

- Directory Processing:
  - The `process_directory` function iterates over audio files in a given directory, classifies them, 
    processes audio data, and saves the corresponding spectrograms.
  - `process_daps_data` function iterates through all specified directories, categorizes them into 
    training and testing sets, and processes files concurrently for efficient data preparation.

"""

import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-GUI rendering
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allowed speakers (class 1)
class_1_speakers = {'F1', 'F7', 'F8', 'M3', 'M6', 'M8'}  # Allowed speakers

# Directories for training and testing data
train_dirs = [
    'clean', 
    'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 'ipad_confroom1',
    'ipad_confroom2', 'ipadflat_confroom1', 'ipadflat_office1', 'ipad_livingroom1',
    'ipad_office1', 'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 'produced'
]
test_dirs = ['iphone_livingroom1']

def load_audio(audio_path, sr=22050, max_duration=120, top_db=30, augment=False):
    """
    Load an audio file, remove silent sections, and split into chunks if longer than max_duration.
    
    Parameters:
    - audio_path: Path to the audio file
    - sr: Sample rate to load the audio at
    - max_duration: Maximum duration of each chunk in seconds
    - top_db: Threshold in decibels for determining silence
    - augment: Flag to enable data augmentation
    
    Returns:
    - List of audio chunks (or empty list if loading fails) and the sample rate
    """
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

    # Augment the audio if enabled
    if augment:
        y_nonsilent = augment_audio(y_nonsilent, sr)

    total_duration = len(y_nonsilent) / sr
    if total_duration > max_duration:
        # Split audio into chunks
        max_samples = int(max_duration * sr)
        audio_chunks = [y_nonsilent[i:i + max_samples] for i in range(0, len(y_nonsilent), max_samples)]
    else:
        # Return entire audio as a single chunk if within max_duration
        audio_chunks = [y_nonsilent]
    
    return audio_chunks, sr

def augment_audio(y, sr):
    """
    Apply random gain (volume change) to the audio.
    
    Parameters:
    - y: Audio data
    - sr: Sample rate
    
    Returns:
    - Augmented audio data
    """
    if np.random.rand() > 0.5: # 50% chance to apply gain augmentation
        gain = np.random.uniform(0.7, 1.3) # Random gain factor between 0.7 and 1.3
        y = y * gain # Apply gain
    return y

def generate_spectrogram(audio_data, sr, output_dir, file_name, segment_idx):
    """
    Generate a mel-spectrogram from audio data and save it as an image.
    
    Parameters:
    - audio_data: The audio data to generate spectrogram from
    - sr: Sample rate of audio
    - output_dir: Directory to save the spectrogram image
    - file_name: Original file name of the audio
    - segment_idx: Index of the audio segment
    
    Returns:
    - Path to saved spectrogram image, or None if an error occurs
    """
    try:
        # Create mel-spectrogram and convert to decibel scale
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        print(f"Error generating spectrogram for {file_name} segment {segment_idx}: {e}")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{file_name}_segment_{segment_idx}_spectrogram.png")
    try:
        # Plot and save the spectrogram
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
    """
    Determine if the speaker in the file is 'allowed' or 'disallowed' based on the speaker ID.
    
    Parameters:
    - file_name: The name of the audio file
    
    Returns:
    - 'allowed' or 'disallowed' based on speaker classification
    """
    speaker_id = file_name.split('_')[0].upper()
    if speaker_id in class_1_speakers:
        return 'allowed'
    else:
        return 'disallowed'

def process_directory(env_path, output_dir, max_duration=120, augment=False):
    """
    Process all .wav files in a given directory, generating spectrograms for each.
    
    Parameters:
    - env_path: Path to the directory with audio files
    - output_dir: Directory to save spectrograms
    - max_duration: Maximum duration of audio segments
    - augment: Flag to enable data augmentation
    """
    for file_name in os.listdir(env_path):
        if file_name.startswith('._') or not file_name.endswith('.wav'):
            continue # Skip non-audio files and hidden files

        audio_path = os.path.join(env_path, file_name)
        class_name = classify_speaker(file_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        # Load audio chunks and sample rate
        audio_chunks, sr = load_audio(audio_path, max_duration=max_duration, augment=augment)

        # Save spectrograms
        for idx, chunk in enumerate(audio_chunks):
            spectrogram_file = generate_spectrogram(chunk, sr, class_output_dir, file_name, idx)
            if spectrogram_file:
                print(f"Spectrogram saved to {spectrogram_file}")

def process_daps_data(daps_dir, output_dir, max_duration=120, augment=False):
    """
    Process directories in the dataset, splitting files into train/test directories and generating spectrograms.
    
    Parameters:
    - daps_dir: Root directory of the dataset
    - output_dir: Directory to save output spectrograms
    - max_duration: Maximum duration of audio segments
    - augment: Flag to enable data augmentation
    """
    files_to_process = []
    for env_dir in os.listdir(daps_dir):
        env_path = os.path.join(daps_dir, env_dir)
        if not os.path.isdir(env_path):
            continue 

        if env_dir in train_dirs:
            output_set_dir = os.path.join(output_dir, 'train')
        elif env_dir in test_dirs:
            output_set_dir = os.path.join(output_dir, 'test')
        else:
            continue
        files_to_process.append((env_path, output_set_dir, max_duration, augment))

    print(f"Processing {len(files_to_process)} directories...")
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_directory, *args) for args in files_to_process]
        with tqdm(total=len(futures), desc="Processing files", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

def main():
    """Main function to process the dataset."""
    daps_dir = './data/daps_data/daps'
    output_dir = './data/spectrograms'
    process_daps_data(daps_dir, output_dir, max_duration=120, augment=True)

if __name__ == "__main__":
    main()
