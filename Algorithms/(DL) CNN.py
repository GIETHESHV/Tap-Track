import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import sys

# Ensure UTF-8 encoding to avoid Unicode errors in Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# Dataset Path
DATASET_PATH = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\All Audio (DL) Dataset"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
BATCH_SIZE = 32
EPOCHS = 30  # Increased for better accuracy
CLICK_DURATION = 0.2  # 200ms per keypress

# Special Character Mapping
LABEL_MAPPING = {
    "Space_20": " ",
    "Enter_20": "\n",
    "Back_20": "⌫",  # Backspace (Handled in prediction)
    "(Backslash)_20": "\\",
    "(Comma)_20": ",",
    "(Dot)_20": ".",
    "(Equal to)_20": "=",
    "(Forward slash)_20": "/",
    "(Hyphen)_20": "-",
    "(Semi colon)_20": ";",
    "(Single string)_20": "'",
    "(Square close)_20": "]",
    "(Square open)_20": "[",
    "0_20": "0", "1_20": "1", "2_20": "2", "3_20": "3", "4_20": "4",
    "5_20": "5", "6_20": "6", "7_20": "7", "8_20": "8", "9_20": "9"
}

# Load Dataset
def load_dataset(dataset_path):
    X, y = [], []
    labels = {}
    
    for i, file in enumerate(os.listdir(dataset_path)):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            label = file.split("_")[0]

            if label not in labels:
                labels[label] = len(labels)  # Assign unique index

            label_index = labels[label]

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                segment_length = int(SAMPLE_RATE * CLICK_DURATION)
                num_segments = len(audio) // segment_length

                for j in range(num_segments):
                    segment = audio[j * segment_length: (j + 1) * segment_length]
                    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_FEATURES)
                    mfccs = np.mean(mfccs.T, axis=0)  # Mean across time axis
                    X.append(mfccs)
                    y.append(label_index)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y), labels

# Load and preprocess data
X, y, labels = load_dataset(DATASET_PATH)
print(f"Loaded {len(X)} samples")
print(f"Labels: {labels}")

# Normalize Features
X = (X - np.mean(X)) / np.std(X)

# Convert Labels to Categorical
y = to_categorical(y, num_classes=len(labels))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for LSTM (Add time steps)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build RNN Model (LSTM)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(40, 1)),  # LSTM for sequential data
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(64, return_sequences=False),  # Second LSTM layer
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')  # Output layer
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save Model
model.save("audio_keypress_model_lstm.h5")

# Prediction Function
def predict_key(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    segment_length = int(SAMPLE_RATE * CLICK_DURATION)
    segments = [audio[i * segment_length: (i + 1) * segment_length] for i in range(len(audio) // segment_length)]

    predicted_key = ""
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Reshape for model

        prediction = model.predict(mfcc)
        predicted_label = np.argmax(prediction)
        
        for key, value in labels.items():
            if value == predicted_label:
                predicted_key += key

    return predicted_key

# Improved Typing Simulation
def predict_article(audio_folder):
    typed_text = ""

    for file in sorted(os.listdir(audio_folder)):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_folder, file)
            try:
                predicted_key = predict_key(file_path)

                # Convert mapped labels back to actual characters
                if predicted_key in LABEL_MAPPING:
                    predicted_key = LABEL_MAPPING[predicted_key]

                # Handle special cases
                if predicted_key == "\n":  # Enter key
                    typed_text += "\n"
                elif predicted_key == "⌫":  # Backspace key
                    typed_text = typed_text[:-1]  # Remove last character
                else:
                    typed_text += predicted_key

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return typed_text

# Predict Article from Test Audio
article_text = predict_article("C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\test audio")
print("Predicted Typed Text:\n", article_text)
