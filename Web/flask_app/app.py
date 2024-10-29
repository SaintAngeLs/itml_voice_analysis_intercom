from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pydub import AudioSegment  # For webm to wav conversion

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Setup file upload folder (absolute path for better file management)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Create 'uploads' folder for saving audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'}  # Allow webm for recording
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists and has the right permissions
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model(os.path.join(BASE_DIR, '../../final_model_2.keras'))

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the audio and extract features (spectrogram)
def preprocess_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=22050)

    # Create the Mel-spectrogram with matching parameters
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to decibel scale (log scale)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    # Resize the spectrogram to 128x128, pad or truncate as necessary
    if mel_spect_db.shape[1] < 128:
        # Pad with zeros if needed
        padding = 128 - mel_spect_db.shape[1]
        mel_spect_db = np.pad(mel_spect_db, ((0, 0), (0, padding)), mode='constant')
    else:
        # Truncate if necessary
        mel_spect_db = mel_spect_db[:, :128]

    # Add batch and channel dimensions to match the input shape of the model
    mel_spect_db_resized = np.expand_dims(mel_spect_db, axis=(0, -1))

    return mel_spect_db_resized


# Convert webm to wav if necessary
def convert_webm_to_wav(webm_file_path, wav_file_path):
    audio = AudioSegment.from_file(webm_file_path, format='webm')
    audio.export(wav_file_path, format='wav')

# Home route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the uploaded file
            file.save(file_path)
            
            # If it's a webm file, convert to wav
            if filename.endswith('.webm'):
                wav_filename = filename.replace('.webm', '.wav')
                wav_file_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
                convert_webm_to_wav(file_path, wav_file_path)
                file_path = wav_file_path  # Use the WAV file for processing
                
        except Exception as e:
            flash(f"Error saving file: {e}")
            return redirect(request.url)

        # Preprocess the audio file and run prediction
        preprocessed_audio = preprocess_audio(file_path)
        predictions = model.predict(preprocessed_audio)
        
        # For binary classification, get predicted probability for "allowed"
        predicted_probability = predictions[0][0]
        predicted_class = 'Allowed' if predicted_probability >= 0.5 else 'Disallowed'
        confidence = predicted_probability * 100 if predicted_class == 'Allowed' else (1 - predicted_probability) * 100

        return render_template('result.html', filename=filename, result=predicted_class, 
                               confidence=confidence)
    else:
        flash('Invalid file format. Please upload a WAV, MP3, or WEBM file.')
        return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5654)
