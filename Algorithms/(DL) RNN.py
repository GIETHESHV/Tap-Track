import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Dataset path
dataset_path = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\All Audio (DL) Dataset"

# Ensure the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: Dataset path '{dataset_path}' does not exist.")
else:
    print("Dataset path exists.")

# Feature Extraction (MFCCs)
def extract_features(audio, sr=22050, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Load dataset and preprocess it
def load_sequence_dataset(directory, sequence_length=10):
    X, y = [], []
    label_map = {}

    files = [f for f in os.listdir(directory) if f.endswith(".wav")]  # Process only .wav files
    if not files:
        raise ValueError("Error: No audio files found in dataset. Check your dataset path and format.")

    for i, file in enumerate(sorted(files)):
        file_path = os.path.join(directory, file)
        
        # Extract label from filename (e.g., "A_20.wav" → "A_20")
        label = file.rsplit(".", 1)[0]
        
        if label not in label_map:
            label_map[label] = len(label_map)  # Assign a unique index

        try:
            audio, sr = librosa.load(file_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract MFCC
            mfcc = np.mean(mfcc, axis=1)  # Flatten MFCC features
            X.append(mfcc)
            y.append(label_map[label])  # Use label index
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    # Ensure the number of labels is divisible by sequence_length
    total_samples = len(y)
    remainder = total_samples % sequence_length
    if remainder != 0:
        print(f"Warning: {remainder} samples will be discarded to ensure divisibility by {sequence_length}.")
        X = X[:total_samples - remainder]
        y = y[:total_samples - remainder]

    # Reshape X and y to fit sequence length
    num_sequences = len(y) // sequence_length
    X = X[:num_sequences * sequence_length]
    y = y[:num_sequences * sequence_length]
    
    X = X.reshape((num_sequences, sequence_length, X.shape[1]))  # Reshape features to include sequence length
    y = y.reshape((num_sequences, sequence_length))  # Reshape labels to include sequence length
    
    return X, y, label_map

# Load sequence dataset
X, y, label_map = load_sequence_dataset(dataset_path, sequence_length=10)

# Ensure X and y have the same number of samples before splitting
if X.shape[0] != y.shape[0]:
    raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]} samples, but y has {y.shape[0]} samples.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Define the RNN Model
def build_rnn_sequence_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
        Bidirectional(GRU(32, return_sequences=True, dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # Output for each time step
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Build the model
model = build_rnn_sequence_model((X_train.shape[1], X_train.shape[2]), len(label_map))

# Train the model and get the history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Display the training and validation accuracy after each epoch
print("\nTraining and validation accuracy after each epoch:")
for epoch in range(30):
    print(f"Epoch {epoch+1}:")
    print(f"  Training accuracy: {history.history['accuracy'][epoch]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][epoch]:.4f}")

# Optionally, display final accuracy after training
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal training accuracy: {train_acc:.4f}")
print(f"Final validation accuracy: {val_acc:.4f}")

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Prediction Function
def predict_sequence(audio_path, model, label_map, sequence_length=10, sample_rate=22050):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Extract features (MFCCs)
    mfcc = extract_features(audio, sr)  # Should be a single vector of features (shape: (n_mfcc,))
    
    # Ensure we have enough data to create a sequence of length `sequence_length`
    mfcc_sequence = np.tile(mfcc, (sequence_length, 1))  # Repeat the same feature vector to match sequence length
    mfcc_sequence = mfcc_sequence.reshape(1, sequence_length, -1)  # Reshape for model input (batch_size, sequence_length, num_features)

    # Make prediction
    prediction = model.predict(mfcc_sequence)

    # Get the predicted labels for each time step in the sequence
    predicted_labels = np.argmax(prediction, axis=-1)

    # Map the predicted indices to the actual labels
    predicted_label_names = [list(label_map.keys())[i] for i in predicted_labels[0]]
    
    return predicted_label_names

# Example Prediction
prediction = predict_sequence("C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\test audio\\5_20.wav", model, label_map)
print(prediction)

