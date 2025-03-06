# Tap-Track: Keypress Audio Classification

## Overview
Tap-Track is a deep learning-based project that classifies keyboard keypresses from recorded audio. The repository implements multiple models to explore different architectures for enhanced accuracy. It supports data preprocessing, feature extraction, training, evaluation, and real-time keypress prediction.

## Features
- **Multiple deep learning models**: CNN, RNN, CNN+BiLSTM, RNN+LSTM, Transformers
- **MFCC-based feature extraction**
- **Residual CNNs & attention mechanisms**
- **Data augmentation & energy-based segmentation**
- **Real-time text prediction from keypress sounds**

## Repository Structure
```
Tap-Track/
│── data/                # Raw and preprocessed datasets
│── models/              # Saved model weights
│── notebooks/           # Jupyter notebooks for training & analysis
│── src/                 # Source code
│   │── preprocess.py    # Data preprocessing and feature extraction
│   │── train.py         # Training script for different models
│   │── predict.py       # Keypress prediction from audio
│── algorithms/          # Different model architectures
│   │── cnn.py           # CNN-based model
│   │── rnn.py           # RNN-based model
│   │── cnn_bilstm.py    # CNN+BiLSTM hybrid model
│   │── rnn_lstm.py      # RNN+LSTM hybrid model
│   │── transformers.py  # Transformer-based model
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

## Setup
### Prerequisites
- Python 3.8+
- TensorFlow / PyTorch
- Required libraries (install using `requirements.txt`)

### Installation
```sh
git clone https://github.com/yourusername/Tap-Track.git
cd Tap-Track
pip install -r requirements.txt
```

## Training Models
Run the respective script to train a specific model:
```sh
python src/train.py --model cnn  # Train CNN model
python src/train.py --model rnn  # Train RNN model
python src/train.py --model cnn_bilstm  # Train CNN+BiLSTM
python src/train.py --model rnn_lstm  # Train RNN+LSTM
python src/train.py --model transformers  # Train Transformer
```

## Prediction
To predict keypresses from audio:
```sh
python src/predict.py --audio path/to/audio.wav
```

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
MIT License

