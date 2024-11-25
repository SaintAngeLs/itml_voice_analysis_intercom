# Voice-Based Intercom System - Machine Learning Project

## Overview
This project focuses on developing a **Voice-Based Intercom System** using **Machine Learning** techniques. By leveraging **Convolutional Neural Networks (CNNs)**, the system classifies voice inputs as either "allowed" or "not allowed" based on user data. The project employs advanced data preprocessing, augmentation, and deep learning methodologies for spectrogram and image analysis.

## Features
- **Audio Preprocessing and Augmentation:** Processes raw audio data into spectrograms, with optional audio augmentation for improved model robustness.
- **CNN-Based Classification:** Utilizes custom CNN models with residual blocks for accurate binary classification of voice patterns.
- **DAPS Dataset Integration:** Integrates the **Device and Produced Speech (DAPS)** dataset for training and validation.
- **TensorBoard Integration:** Provides visualization of training metrics and performance insights through TensorBoard.
- **Modular Pipeline:** Includes modules for data preprocessing, model training, evaluation, and deployment.

## Dataset: DAPS (Device and Produced Speech)
The **DAPS Dataset** is used for training and evaluating the model. This dataset includes recordings of speech in various environments, providing a robust dataset for real-world audio classification tasks.

### Key Details:
- **Dataset Composition:** 
  - 15 versions of audio (3 professional, 12 consumer device/real-world environments).
  - Approximately 4.5 hours of data per version, with 20 unique speakers.
- **Applications:** 
  - Speech enhancement, voice conversion, and audio classification.

**Dataset Source:** [DAPS Dataset on Zenodo](https://zenodo.org/records/4660670)

### Citation:
If you use the DAPS dataset, please cite:
> Gautham J. Mysore, "Can We Automatically Transform Speech Recorded on Common Consumer Devices in Real-World Environments into Professional Production Quality Speech? - A Dataset, Insights, and Challenges," in the IEEE Signal Processing Letters, Vol. 22, No. 8, August 2015.

---

## Data Processing
The data pipeline includes:
1. **Audio Loading and Preprocessing:**
   - Non-silent intervals are extracted using `librosa`.
   - Audio is split into manageable chunks for efficient training.
2. **Audio Augmentation:**
   - Random gain adjustments for training robustness.
3. **Spectrogram Generation:**
   - Converts audio chunks into mel-spectrogram images using `librosa`.
   - Saves spectrograms as `.png` files for model training.
4. **Classification:**
   - Audio files are classified as "allowed" or "disallowed" based on pre-defined speaker IDs.

### Code Overview:
- **`AudioProcessor`**: Handles audio loading, preprocessing, and augmentation.
- **`SpectrogramGenerator`**: Converts audio chunks to spectrograms.
- **`DatasetProcessor`**: Manages dataset-wide preprocessing and spectrogram generation.

---

## Model
The model is a **CNN-based architecture** with residual blocks for high performance in binary classification tasks.

### Key Features:
- **Residual Blocks:** Ensures deep feature extraction while mitigating vanishing gradients.
- **Global Average Pooling:** Reduces overfitting by summarizing spatial features.
- **Regularization:** L2 regularization and dropout are used to improve generalization.
- **Output Activation:** Uses sigmoid activation for binary classification.

### Architecture:
1. Initial Convolutional Block
2. Residual Blocks:
   - Three stages with increasing filter sizes: 64, 128, and 256.
3. Fully Connected Layers:
   - Dense layers with dropout for robust feature learning.
4. Output Layer:
   - Sigmoid activation for binary classification.

---

## Training
### Training Pipeline:
1. **Data Loading:**
   - Preprocessed spectrograms are loaded into the model pipeline.
2. **Train/Validation/Test Split:**
   - Data is split into 70% training, 20% validation, and 10% testing.
3. **Training:**
   - The model is trained using binary cross-entropy loss and the Adam optimizer.
   - Early stopping and learning rate scheduling improve convergence.
4. **Evaluation:**
   - Test set performance is evaluated for accuracy and loss.

### Hyperparameters:
- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.0001 with decay.

### Callbacks:
- **Early Stopping:** Stops training if validation loss stops improving.
- **Model Checkpoint:** Saves the best model based on validation performance.
- **TensorBoard Logging:** Tracks training and validation metrics.

---

## TensorBoard Integration
TensorBoard is used for:
- Monitoring training and validation loss.
- Visualizing training accuracy.
- Understanding the impact of data augmentation and hyperparameter tuning.

### Usage:
To view TensorBoard logs:
```bash
tensorboard --logdir=./logs
```
Access TensorBoard at `http://localhost:6006`.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd itml_voice_analysis_intercom
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```

4. Download and organize the DAPS dataset:
   ```
   data/
   ├── daps_data/
       ├── daps/
           ├── clean/
           ├── cleanraw/
           ├── [Other DAPS subdirectories]
   ```

---

## Usage
### Data Processing:
Preprocess the audio data and generate spectrograms:
```bash
python src/data_preprocessing.py
```

### Training:
Train the CNN model:
```bash
python src/train.py
```

### Evaluation:
Evaluate the model on the test set:
```bash
python src/evaluate.py
```

### Web Application:
Run the Flask web application:
```bash
cd Web/flask_app
python app.py
```
Access the app at `http://127.0.0.1:5000`.

---

## Project Structure
```
.
├── data/                   # Raw and preprocessed datasets
├── src/                    # Source code for training, evaluation, and utilities
├── models/                 # Saved models and label encoders
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── train_results/          # Training results and checkpoints
├── logs/                   # Logs for training and evaluation
├── Web/                    # Flask-based web application for model deployment
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## Future Work
- Incorporate additional datasets to improve generalization.
- Deploy the model on edge devices for real-time voice recognition.
- Add advanced audio augmentation techniques to improve model robustness.

## License
This project is licensed under [MIT License](LICENSE). Refer to the DAPS dataset's license for terms of use.
