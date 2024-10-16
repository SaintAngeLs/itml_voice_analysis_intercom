from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Setup file upload folder (absolute path for better file management)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Create 'uploads' folder for saving audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists and has the right permissions
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model and label encoder
model = load_model(os.path.join(BASE_DIR, '../../models/cnn_model.keras'))
label_encoder = np.load(os.path.join(BASE_DIR, '../../models/label_encoder.npy'), allow_pickle=True)

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the audio and extract features (spectrogram)
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Create the Mel-spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Resize the spectrogram to the shape expected by the model (128x128)
    mel_spect_db_resized = np.resize(mel_spect_db, (128, 128))
    
    # Add batch and channel dimensions to match the input shape of the model
    mel_spect_db_resized = np.expand_dims(mel_spect_db_resized, axis=(0, -1))
    
    return mel_spect_db_resized

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
            # Save the file
            file.save(file_path)
        except Exception as e:
            flash(f"Error saving file: {e}")
            return redirect(request.url)

        # Preprocess the audio file and run prediction
        preprocessed_audio = preprocess_audio(file_path)
        predictions = model.predict(preprocessed_audio)
        
        predicted_class = np.argmax(predictions, axis=1)
        predicted_user = label_encoder[predicted_class[0]]  # Map the predicted class to the user
        
        confidence = np.max(predictions) * 100  # Get confidence percentage

        return render_template('result.html', filename=filename, result='Allowed', 
                               confidence=confidence, identified_user=predicted_user)
    else:
        flash('Invalid file format. Please upload a WAV or MP3 file.')
        return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5654)