import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(audio_path, sr=22050, max_duration=10, top_db=30):
    """Load audio file, remove silence, and return segments if the audio is longer than max_duration seconds."""
    y, sr = librosa.load(audio_path, sr=sr)

    # Remove silent parts of the audio
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    y_nonsilent = np.concatenate([y[start:end] for start, end in non_silent_intervals])

    max_samples = int(max_duration * sr)  # Maximum samples for the duration
    
    # If the audio is longer than max_duration, split it into chunks
    if len(y_nonsilent) > max_samples:
        audio_chunks = [y_nonsilent[i:i + max_samples] for i in range(0, len(y_nonsilent), max_samples)]
    else:
        audio_chunks = [y_nonsilent]  # If it's shorter than max_duration, no chunking

    return audio_chunks, sr

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

def process_audio_files_by_user(input_dir, output_dir, class_name, max_duration=5):
    """Process all audio files in the input directory for a specific class (allowed)."""
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
                    
                    # Load audio, remove silence, and split into chunks if necessary
                    audio_chunks, sr = load_audio(audio_path, max_duration=max_duration)
                    
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

def process_test_voices(test_dir, allowed_files, output_dir, max_duration=5):
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

            # Load audio, remove silence, and split into chunks if necessary
            audio_chunks, sr = load_audio(audio_path, max_duration=max_duration)

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
    
    # Process allowed voices (save spectrograms and MFCCs in "allowed", categorized by user)
    process_audio_files_by_user(allowed_dir, output_dir, 'allowed')
    
    # Process test voices (classify as "allowed" or "disallowed")
    process_test_voices(test_dir, allowed_files, output_dir)

if __name__ == "__main__":
    main()
