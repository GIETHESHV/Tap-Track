import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical

# Constants
DATASET_PATH = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\All Audio (DL) Dataset"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40          # Number of MFCC coefficients
TIME_STEPS = 64 #32         # Desired number of time frames per keypress segment
BATCH_SIZE = 32
EPOCHS = 50 #30
CLICK_DURATION = 0.2        # Each keypress duration in seconds

# Load dataset with full MFCC sequences (without averaging)
def load_dataset(dataset_path):
    X, y = [], []
    labels = {}
    
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            label = file.split("_")[0]  # Extract label from filename
            
            if label not in labels:
                labels[label] = len(labels)
            label_index = labels[label]

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                segment_length = int(SAMPLE_RATE * CLICK_DURATION)
                num_segments = len(audio) // segment_length

                for j in range(num_segments):
                    segment = audio[j * segment_length: (j + 1) * segment_length]
                    # Extract MFCCs; shape: (MFCC_FEATURES, T)
                    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_FEATURES)
                    
                    # Normalize the MFCCs
                    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
                    
                    # Pad (or truncate) along the time axis (axis=1) to ensure TIME_STEPS
                    if mfccs.shape[1] < TIME_STEPS:
                        mfccs = np.pad(mfccs, ((0, 0), (0, TIME_STEPS - mfccs.shape[1])), mode="constant")
                    else:
                        mfccs = mfccs[:, :TIME_STEPS]
                    
                    # Transpose so that shape becomes (TIME_STEPS, MFCC_FEATURES)
                    X.append(mfccs.T)
                    y.append(label_index)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y), labels

# Load and process the dataset
X, y, labels = load_dataset(DATASET_PATH)
print(f"Loaded {len(X)} samples")
print(f"Labels: {labels}")

# One-hot encode labels
y = to_categorical(y, num_classes=len(labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build RNN Model with Bi-LSTM
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, activation='relu'), input_shape=(TIME_STEPS, MFCC_FEATURES)),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='relu')),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save("audio_keypress_rnn_model.h5")

# Prediction Function
def predict_key(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    segment_length = int(SAMPLE_RATE * CLICK_DURATION)
    segments = [audio[i * segment_length: (i + 1) * segment_length] for i in range(len(audio) // segment_length)]
    
    predicted_keys = []
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # Pad/truncate along the time axis (axis=1) to ensure TIME_STEPS
        if mfcc.shape[1] < TIME_STEPS:
            mfcc = np.pad(mfcc, ((0, 0), (0, TIME_STEPS - mfcc.shape[1])), mode="constant")
        else:
            mfcc = mfcc[:, :TIME_STEPS]
        
        # Transpose to shape (TIME_STEPS, MFCC_FEATURES) and add batch dimension
        mfcc = mfcc.T[np.newaxis, ...]
        
        prediction = model.predict(mfcc)
        predicted_label = np.argmax(prediction)
        
        # Map predicted index to the corresponding key label
        for key, value in labels.items():
            if value == predicted_label:
                predicted_keys.append(key)
    return predicted_keys

# Prediction for a full article (handling Enter and Backspace)
def predict_article(audio_folder):
    typed_text = []
    for file in sorted(os.listdir(audio_folder)):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_folder, file)
            try:
                predicted_keys = predict_key(file_path)
                for key in predicted_keys:
                    if key == "Enter_20":
                        typed_text.append("\n")
                    elif key == "Back_20" and typed_text:
                        typed_text.pop()  # Remove the last character
                    else:
                        typed_text.append(key.replace("_20", ""))  # Remove the _20 suffix if present
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return "".join(typed_text)

# Test the model on a sample folder of keypress audio files
article_text = predict_article("C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\test audio")
print("Predicted Typed Text:\n", article_text)
