import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Transformer-based model
class KeypressTransformer(nn.Module):
    def __init__(self, input_dim=40, embed_dim=64, num_heads=8, num_classes=40):
        super(KeypressTransformer, self).__init__()

        print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, embed_dim % num_heads: {embed_dim % num_heads}")
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=4
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        print(f"DEBUG: Before Embedding, x.shape: {x.shape}")  # Add this line
        x = self.embedding(x)
        print(f"DEBUG: After Embedding, x.shape: {x.shape}")  # Add this line
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Mean over sequence dimension
        x = self.fc(x)
        return x

# Audio Preprocessing
def preprocess_audio(audio, sample_rate, max_len=1100):
    transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, n_mels=40)
    mel_spec = transform(audio)
    mel_spec = torch.log(mel_spec + 1e-9)  # Convert to log scale

    # Ensure consistent length
    current_len = mel_spec.shape[2]
    if current_len < max_len:
        mel_spec = F.pad(mel_spec, (0, max_len - current_len))  # Pad shorter spectrograms
    else:
        mel_spec = mel_spec[:, :, :max_len]  # Trim longer spectrograms

    return mel_spec

# Load dataset
def load_dataset(data_dir, max_len=1100):
    dataset = []
    labels = []
    class_mapping = {}
    class_index = 0  

    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            label = filename.split("_")[0]  # Extract label

            if label not in class_mapping:
                class_mapping[label] = class_index
                class_index += 1

            waveform, sample_rate = torchaudio.load(os.path.join(data_dir, filename))
            processed_audio = preprocess_audio(waveform, sample_rate, max_len)
            dataset.append(processed_audio)
            labels.append(class_mapping[label])

    dataset = torch.stack(dataset)
    labels = torch.tensor(labels)

    # Reshape dataset to (batch_size, seq_len, input_dim)
    dataset = dataset.squeeze(1).permute(0, 2, 1)  # (batch_size, seq_len, input_dim)

    return dataset, labels, class_mapping

# Training function
def train_model(model, train_data, train_labels, epochs=10, lr=0.001, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save and Load Model
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict Keypress from Audio
def predict_keypress(model, audio_path, class_mapping):
    print(f"Loading test audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        print("Error: Audio file not found!")
        return None

    waveform, sample_rate = torchaudio.load(audio_path)
    processed_audio = preprocess_audio(waveform, sample_rate).squeeze(1).permute(1, 0)

    with torch.no_grad():
        output = model(processed_audio.unsqueeze(0).to(device))
    
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_label = {v: k for k, v in class_mapping.items()}.get(predicted_class, "Unknown Key")
    
    print("Final Predicted Keypress:", predicted_label)
    return predicted_label

# Main script
if __name__ == "__main__":
    train_data_dir = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\All Audio (DL) Dataset"
    model_path = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\PY Code for DL model\\(DL) Transformers.pth"
    test_audio_path = "C:\\Users\\GIETHU\\OneDrive\\Desktop\\Internship NITT\\Project\\DL Model\\Audio Dataset\\test_audio.wav"

    # Load dataset
    print("Loading dataset...")
    train_data, train_labels, class_mapping = load_dataset(train_data_dir)
    print("Dataset Loaded. Classes:", class_mapping)

    # Initialize model
    input_dim = train_data.shape[-1]  # Feature size
    num_classes = len(class_mapping)

    print(f"DEBUG: input_dim={input_dim}, embed_dim=64, num_heads=8, num_classes={num_classes}")

    model = KeypressTransformer(input_dim=input_dim, embed_dim=64, num_heads=8, num_classes=num_classes)

    # Train model
    print("Training model...")
    train_model(model, train_data, train_labels)

    # Save model
    print("Saving model...")
    save_model(model, model_path)
    print("Model saved successfully!")

    # Load model
    print("Loading trained model...")
    model = load_model(model, model_path)
    print("Model Loaded Successfully!")

    # Predict keypress
    print("Running prediction...")
    predicted_key = predict_keypress(model, test_audio_path, class_mapping)
    print("Predicted Keypress:", predicted_key)
