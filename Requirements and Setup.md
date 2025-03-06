This repository contains multiple deep learning models for classifying keypresses based on audio recordings. The models implemented include:

- **CNN (Convolutional Neural Network)**
- **RNN (Recurrent Neural Network)**
- **CNN + BiLSTM (Bidirectional LSTM with CNN features)**
- **RNN + LSTM (Combination of RNN and LSTM layers)**
- **Transformer-based Model**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/keypress-audio-classification.git
   cd keypress-audio-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Ensure that your dataset is structured as follows:
```
Dataset/
├── class_1/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── class_2/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
...
```
Update `data_dir` in your training script to match your dataset location.

## Training Models
Each model can be trained using the corresponding script:

- **CNN Model**: `train_cnn.py`
- **RNN Model**: `train_rnn.py`
- **CNN + BiLSTM Model**: `train_cnn_bilstm.py`
- **RNN + LSTM Model**: `train_rnn_lstm.py`
- **Transformer Model**: `train_transformer.py`

Run the desired training script:
```bash
python train_cnn.py
```

## Model Evaluation
After training, you can evaluate models using:
```bash
python evaluate_model.py --model transformer --weights path/to/model.pth
```

## Predict Keypress from Audio
To classify a new audio sample:
```bash
python predict.py --model transformer --audio path/to/audio.wav
```

## Saved Model Weights
The trained model weights are stored in the `saved_models/` directory:
```
saved_models/
├── cnn_model.pth
├── rnn_model.pth
├── cnn_bilstm_model.pth
├── rnn_lstm_model.pth
├── transformer_model.pth
```

## Requirements
Below is a table listing required dependencies:

| Package       | Version |
|--------------|---------|
| torch        | >=1.10  |
| torchaudio   | >=0.10  |
| numpy        | >=1.21  |
| pandas       | >=1.3   |
| matplotlib   | >=3.4   |
| librosa      | >=0.8   |
| scipy        | >=1.7   |
| tqdm        | >=4.62  |

## Contributing
Feel free to contribute by submitting pull requests or opening issues.

## License
This project is licensed under the MIT License.

