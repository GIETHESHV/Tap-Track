import os
import numpy as np
import librosa
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    Add, Flatten, Bidirectional, LSTM, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# ==================== Constants ====================
DATASET_PATH = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\All Audio (DL) Dataset"
TEST_AUDIO_FOLDER = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\test audio"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40          # Number of MFCC coefficients
TIME_STEPS = 32             # Fixed number of time frames per keypress segment
CLICK_DURATION = 0.2        # Duration of each keypress in seconds
BATCH_SIZE = 32
EPOCHS = 70                 # More epochs for deeper training
AUGMENT_PROB = 0.7          # Higher probability for augmentation

# ==================== Advanced Data Augmentation ====================
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift(audio, sr, n_steps=2):
    try:
        return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio

def time_stretch(audio, rate=1.1):
    try:
        return librosa.effects.time_stretch(audio, rate)
    except Exception as e:
        print(f"Time stretch error: {e}")
        return audio

def dynamic_range_compression(audio, compression_factor=0.8):
    return np.tanh(compression_factor * audio)

def augment_audio(audio, sr):
    try:
        if random.random() < AUGMENT_PROB:
            audio = add_noise(audio, noise_factor=random.uniform(0.002, 0.01))
        if random.random() < AUGMENT_PROB:
            audio = pitch_shift(audio, sr, n_steps=random.choice([-2, -1, 1, 2]))
        if random.random() < AUGMENT_PROB:
            audio = time_stretch(audio, rate=random.choice([0.7, 1.0]))
        if random.random() < AUGMENT_PROB:
            audio = dynamic_range_compression(audio, compression_factor=random.uniform(0.7, 1.0))
    except Exception as e:
        print(f"Augmentation error: {e}")
    return audio

# ==================== Feature Extraction ====================
def extract_mfcc(audio, sr):
    try:
        # Compute MFCCs: shape (MFCC_FEATURES, T)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_FEATURES)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)  # Normalize
        # Pad or truncate along time axis (axis=1) to ensure TIME_STEPS frames
        if mfccs.shape[1] < TIME_STEPS:
            pad_width = TIME_STEPS - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :TIME_STEPS]
        # Transpose to shape: (TIME_STEPS, MFCC_FEATURES)
        return mfccs.T
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros((TIME_STEPS, MFCC_FEATURES))

# ==================== Dataset Loader ====================
def load_dataset(dataset_path, augment=False):
    X, y = [], []
    labels = {}
    
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            try:
                # Expecting filenames like "A_20.wav", "B_20.wav", etc.
                label = file.split("_")[0]
                if label not in labels:
                    labels[label] = len(labels)
                label_index = labels[label]
                
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                if augment:
                    audio = augment_audio(audio, sr)
                segment_length = int(SAMPLE_RATE * CLICK_DURATION)
                num_segments = len(audio) // segment_length
                if num_segments < 1:
                    continue
                for j in range(num_segments):
                    segment = audio[j * segment_length: (j + 1) * segment_length]
                    mfcc_features = extract_mfcc(segment, sr)  # (TIME_STEPS, MFCC_FEATURES)
                    if mfcc_features.shape != (TIME_STEPS, MFCC_FEATURES):
                        print(f"Warning: Unexpected MFCC shape in file {file}. Expected {(TIME_STEPS, MFCC_FEATURES)}, got {mfcc_features.shape}")
                    X.append(mfcc_features)
                    y.append(label_index)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return np.array(X), np.array(y), labels

# ==================== Load and Prepare Data ====================
print("Loading dataset without augmentation...")
X_all, y_all, labels = load_dataset(DATASET_PATH, augment=False)
print(f"Loaded {len(X_all)} samples")
print(f"Label Mapping: {labels}")

try:
    y_all = to_categorical(y_all, num_classes=len(labels))
except Exception as e:
    print(f"Error during one-hot encoding: {e}")

try:
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Train-test split error: {e}")

print("Loading augmented training data...")
X_train_aug, y_train_aug, _ = load_dataset(DATASET_PATH, augment=True)
try:
    y_train_aug = to_categorical(y_train_aug, num_classes=len(labels))
except Exception as e:
    print(f"Error during one-hot encoding of augmented labels: {e}")

# Check dimensions before concatenation and then combine
if y_train.ndim != y_train_aug.ndim:
    print("Dimension mismatch between original and augmented labels. Adjusting...")
    y_train_aug = np.expand_dims(y_train_aug, axis=-1)
try:
    X_train = np.concatenate((X_train, X_train_aug), axis=0)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)
except Exception as e:
    print(f"Error concatenating training data: {e}")

# ==================== Model Architecture ====================
# Define a residual convolutional block (with fixed pooling on the shortcut branch)
def residual_conv_block(x, filters, kernel_size=3, pool_size=2):
    shortcut = x
    x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    # Adjust shortcut: apply a 1x1 convolution and then same pooling to match dimensions
    shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    shortcut = MaxPooling1D(pool_size=pool_size)(shortcut)
    x = Add()([x, shortcut])
    return x

# Define a simple attention layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Build the model using the Functional API
input_layer = Input(shape=(TIME_STEPS, MFCC_FEATURES))
# Initial Conv1D layer
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
# Residual convolutional block
x = residual_conv_block(x, 64, kernel_size=3, pool_size=2)
x = Dropout(0.3)(x)
# Bidirectional LSTM layers
x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh'))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(x)
x = Dropout(0.3)(x)
# Attention mechanism
x = Attention()(x)
# Dense layers for classification
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==================== Callbacks ====================
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint("best_audio_keypress_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# ==================== Model Training ====================
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )
except Exception as e:
    print(f"Error during model training: {e}")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the final model
model.save("audio_keypress_model_strong.h5")

# ==================== Prediction Functions ====================
def segment_audio(audio, sr, click_duration, energy_threshold=0.005):
    segment_length = int(sr * click_duration)
    segments = []
    num_segments = len(audio) // segment_length
    for i in range(num_segments):
        seg = audio[i * segment_length: (i + 1) * segment_length]
        # Calculate mean absolute energy of the segment
        if np.mean(np.abs(seg)) >= energy_threshold:
            segments.append(seg)
    return segments

def predict_key(audio_path, energy_threshold=0.01):
    """
    Loads an audio file, segments it using energy filtering, and predicts the key for each segment.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []
    
    # Use the refined segmentation with energy filtering
    segments = segment_audio(audio, SAMPLE_RATE, CLICK_DURATION, energy_threshold)
    
    def predict_key(audio_path, energy_threshold=0.01):
        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return []
    
    # Segment the audio with energy filtering
    segments = segment_audio(audio, SAMPLE_RATE, CLICK_DURATION, energy_threshold)
    
    predicted_keys = []
    for segment in segments:
        try:
            mfcc = extract_mfcc(segment, SAMPLE_RATE)
            mfcc = mfcc[np.newaxis, ...]  # shape: (1, TIME_STEPS, MFCC_FEATURES)
            prediction = model.predict(mfcc)  # Assumes your trained model is loaded as 'model'
            predicted_index = np.argmax(prediction)
            # Map the predicted index to its key label.
            for key, index in labels.items():  # Assumes your label dictionary is available as 'labels'
                if index == predicted_index:
                    predicted_keys.append(key)
                    break
        except Exception as e:
            print(f"Error in prediction for a segment: {e}")
    return predicted_keys

def predict_article(audio_folder, energy_threshold=0.01, merge_consecutive=False):
    typed_text = []
    # Loop over each test audio file in the folder
    for file in sorted(os.listdir(audio_folder)):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_folder, file)
            try:
                predicted_keys = predict_key(file_path, energy_threshold)
                if merge_consecutive:
                    # Merge consecutive identical keys
                    merged_keys = []
                    for key in predicted_keys:
                        if merged_keys and merged_keys[-1] == key:
                            continue
                        merged_keys.append(key)
                    predicted_keys = merged_keys
                
                # Process predicted keys to handle special keys if needed
                for key in predicted_keys:
                    if key == "Enter_20":
                        typed_text.append("\n")
                    elif key == "Back_20" and typed_text:
                        typed_text.pop()  # Remove last character
                    else:
                        # Remove the suffix if present (e.g., "B_20" becomes "B")
                        typed_text.append(key.replace("_20", ""))
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return "".join(typed_text)

# ==================== Test Predictions ====================
try:
    article_text = predict_article(TEST_AUDIO_FOLDER, energy_threshold=0.01, merge_consecutive=False)
    print("Predicted Typed Text:\n", article_text)
except Exception as e:
    print(f"Error during article prediction: {e}")
