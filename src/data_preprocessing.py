import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import random

def load_audio(audio_path, sr=22050, max_duration=10, top_db=30, augment=False):
    """Load audio file, remove silence and noise, and return segments if the audio is longer than max_duration seconds."""
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Remove silent parts of the audio
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    # Apply noise reduction
    y_denoised = reduce_noise(y_nonsilent, sr)

    # Optionally augment the audio data
    if augment:
        y_denoised = augment_audio(y_denoised, sr)

    max_samples = int(max_duration * sr)  # Maximum samples for the duration

    # If the audio is longer than max_duration, split it into chunks
    if len(y_denoised) > max_samples:
        audio_chunks = [y_denoised[i:i + max_samples] for i in range(0, len(y_denoised), max_samples)]
    else:
        audio_chunks = [y_denoised]  # If it's shorter than max_duration, no chunking

    return audio_chunks, sr

def reduce_noise(y, sr):
    """Reduce noise from an audio signal using spectral gating techniques."""
    # Estimate noise from the beginning part of the signal (first 1 second)
    noise_sample = y[:sr]
    
    # Apply noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
    
    return y_denoised

def augment_audio(y, sr):
    """Apply random augmentations to the audio data."""
    
    # Random Volume Control
    if random.random() > 0.5:
        gain = random.uniform(0.7, 1.3)  # Random volume change between 70% to 130%
        y = y * gain

    # Time Stretching (Speed Perturbation) - Corrected time_stretch usage
    if random.random() > 0.5:
        rate = random.uniform(0.8, 1.2)  # Stretch/compress time between 80% to 120%
        y = librosa.effects.time_stretch(y, rate=rate)  # Corrected to pass `rate` as a keyword argument

    # Pitch Shifting
    if random.random() > 0.5:
        n_steps = random.randint(-4, 4)  # Shift pitch between -4 and +4 semitones
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Adding White Noise
    if random.random() > 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)

    # Time Masking
    if random.random() > 0.5:
        y = apply_time_masking(y, sr)

    # Frequency Masking
    if random.random() > 0.5:
        y = apply_frequency_masking(y, sr)

    # Reverberation (adding slight echo)
    if random.random() > 0.5:
        y = add_reverb(y, sr)

    return y

def apply_time_masking(y, sr, max_mask_pct=0.1):
    """Randomly mask some part of the audio data by zeroing out segments of time."""
    max_mask_len = int(max_mask_pct * len(y))
    mask_start = random.randint(0, len(y) - max_mask_len)
    y[mask_start:mask_start + max_mask_len] = 0
    return y

def apply_frequency_masking(y, sr, max_mask_pct=0.1):
    """Apply frequency masking in the frequency domain to simulate dropout of certain frequency ranges."""
    # Generate the spectrogram
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    
    num_freq_bins = S_db.shape[0]
    mask_start = random.randint(0, int(num_freq_bins * (1 - max_mask_pct)))
    mask_end = mask_start + int(num_freq_bins * max_mask_pct)
    
    S_db[mask_start:mask_end, :] = 0  # Apply mask
    
    # Inverse transformation back to time domain
    y_masked = librosa.istft(librosa.db_to_amplitude(S_db))
    
    return y_masked

def add_reverb(y, sr, reverb_factor=0.2):
    """Add reverberation by convolving the audio signal with an impulse response."""
    reverb = np.convolve(y, np.random.randn(int(reverb_factor * sr)), mode='full')
    reverb = reverb[:len(y)]  # Keep the same length as original audio
    return reverb

def generate_spectrogram(audio_data, sr, output_dir, file_name, segment_idx):
    """Generate a mel-spectrogram for each audio segment and save it."""
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add segment index to the filename for each segment
    output_file = os.path.join(output_dir, f"{file_name}_segment_{segment_idx}_spectrogram.png")
    
    # Generate and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file

def generate_mfcc(audio_data, sr, output_dir, file_name, segment_idx):
    """Generate MFCC and save it as an image."""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add segment index to the filename for each segment
    output_file = os.path.join(output_dir, f"{file_name}_segment_{segment_idx}_mfcc.png")
    
    # Generate and save the MFCC plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file

def process_audio_files_by_user(input_dir, output_dir, class_name, max_duration=5, augment=False):
    """Process all audio files in the input directory for a specific class (allowed), with optional augmentation."""
    class_output_dir = os.path.join(output_dir, class_name)
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    
    # Traverse each user's directory inside the input_dir
    for user_dir in os.listdir(input_dir):
        user_path = os.path.join(input_dir, user_dir)
        if os.path.isdir(user_path):  # Ensure it's a directory
            print(f"Processing user: {user_dir}")
            
            # Process each file in the user's directory
            for file_name in os.listdir(user_path):
                if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                    audio_path = os.path.join(user_path, file_name)
                    print(f"Processing {audio_path} for class {class_name} (user: {user_dir})...")
                    
                    # Use user-specific output directory
                    user_output_dir = os.path.join(class_output_dir, user_dir)
                    
                    # Load audio, remove silence, reduce noise, and split into chunks if necessary
                    audio_chunks, sr = load_audio(audio_path, max_duration=max_duration, augment=augment)
                    
                    # Generate spectrogram and MFCC for each chunk
                    for idx, chunk in enumerate(audio_chunks):
                        spectrogram_file = generate_spectrogram(chunk, sr, user_output_dir, file_name, idx)
                        mfcc_file = generate_mfcc(chunk, sr, user_output_dir, file_name, idx)
                        print(f"Spectrogram for segment {idx} saved to {spectrogram_file}")
                        print(f"MFCC for segment {idx} saved to {mfcc_file}")

def get_allowed_files(allowed_dir):
    """Get the list of allowed voice files from the allowed directory."""
    allowed_files = set()
    if os.path.exists(allowed_dir):
        for user_dir in os.listdir(allowed_dir):
            user_path = os.path.join(allowed_dir, user_dir)
            if os.path.isdir(user_path):  # Only process directories
                for file_name in os.listdir(user_path):
                    if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                        allowed_files.add(file_name)
    return allowed_files

def process_test_voices(test_dir, allowed_files, output_dir, max_duration=5, augment=False):
    """Process test voices and classify them as allowed or disallowed based on allowed_files."""
    allowed_output_dir = os.path.join(output_dir, 'allowed')
    disallowed_output_dir = os.path.join(output_dir, 'disallowed')

    if not os.path.exists(test_dir):
        print(f"Input directory {test_dir} does not exist.")
        return
    
    for file_name in os.listdir(test_dir):
        if file_name.endswith('.wav') or file_name.endswith('.mp3'):
            audio_path = os.path.join(test_dir, file_name)
            
            if file_name in allowed_files:
                class_name = 'allowed'
                output_folder = allowed_output_dir
            else:
                class_name = 'disallowed'
                output_folder = disallowed_output_dir

            print(f"Processing {audio_path} as {class_name}...")

            # Load audio, remove silence, reduce noise, and split into chunks if necessary, with optional augmentation
            audio_chunks, sr = load_audio(audio_path, max_duration=max_duration, augment=augment)

            # Generate spectrogram and MFCC for each chunk
            for idx, chunk in enumerate(audio_chunks):
                spectrogram_file = generate_spectrogram(chunk, sr, output_folder, file_name, idx)
                mfcc_file = generate_mfcc(chunk, sr, output_folder, file_name, idx)
                print(f"Spectrogram for segment {idx} saved to {spectrogram_file}")
                print(f"MFCC for segment {idx} saved to {mfcc_file}")

def main():
    # Paths to the voice folders
    allowed_dir = './data/allowed'  # Directory for allowed voices (user-specific subdirectories)
    test_dir = './data/test_voices'  # Directory for test voices (for checking if allowed)
    output_dir = './data/spectrograms'  # Where spectrograms and MFCCs will be saved
    
    # Get list of allowed voice files
    allowed_files = get_allowed_files(allowed_dir)
    
    # Process allowed voices (save spectrograms and MFCCs in "allowed", categorized by user, with augmentations)
    process_audio_files_by_user(allowed_dir, output_dir, 'allowed', augment=True)
    
    # Process test voices (classify as "allowed" or "disallowed", with augmentations)
    process_test_voices(test_dir, allowed_files, output_dir, augment=True)

if __name__ == "__main__":
    main()
