import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(audio_path, sr=22050):
    """Load audio file and return the audio time series and sampling rate."""
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr

def generate_spectrogram(audio_path, output_dir, sr=22050, n_mels=128):
    """Generate a mel-spectrogram and save it as an image with a '_spectrogram' suffix."""
    y, sr = load_audio(audio_path, sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the spectrogram image path with the '_spectrogram' suffix
    file_name = os.path.basename(audio_path).replace('.wav', '_spectrogram.png').replace('.mp3', '_spectrogram.png')
    output_file = os.path.join(output_dir, file_name)
    
    # Generate and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def process_audio_files_by_user(input_dir, output_dir, class_name):
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
                    output_file = generate_spectrogram(audio_path, user_output_dir)
                    print(f"Spectrogram saved to {output_file}")

def process_test_voices(test_dir, allowed_files, output_dir):
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
            output_file = generate_spectrogram(audio_path, output_folder)
            print(f"Spectrogram saved to {output_file}")

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

def main():
    # Paths to the voice folders
    allowed_dir = './data/allowed'  # Directory for allowed voices (user-specific subdirectories)
    test_dir = './data/test_voices'  # Directory for test voices (for checking if allowed)
    output_dir = './data/spectrograms'  # Where spectrograms will be saved
    
    # Get list of allowed voice files
    allowed_files = get_allowed_files(allowed_dir)
    
    # Process allowed voices (save spectrograms in "allowed", categorized by user)
    process_audio_files_by_user(allowed_dir, output_dir, 'allowed')
    
    # Process test voices (classify as "allowed" or "disallowed")
    process_test_voices(test_dir, allowed_files, output_dir)

if __name__ == "__main__":
    main()
