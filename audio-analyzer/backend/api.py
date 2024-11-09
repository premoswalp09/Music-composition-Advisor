import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import BertTokenizer
from torch.nn.functional import pad
import logging
import sys
from audioTrain import AudioDataset, AudioClassifier, predict_genre_from_audio, train_audio_model
from textTrain import TextDataset, TextClassifier, predict_sentiment_and_genre, train_text_model, make_predict
from audio_text_Train import load_encoders, predict_complete_lyrics
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration parameters
MFCC_N = 64
TARGET_LENGTH = 2556  # Set this to match the model's expected input width for audio analysis
MAX_LENGTH_TEXT = 500

# Define your model architecture
class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 1)  # 512 because bidirectional LSTM (256*2)

    def forward(self, x, attention_mask=None):
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        output = self.fc(dropped)
        return output


app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# Load pre-trained models
def load_text_model(model_path):
    logger.info(f"Loading model from {model_path}")
    try:
        # Create model instance
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # Load encoders
        genre_encoder, sentiment_encoder = load_encoders()

        # Load models
        num_genres = len(genre_encoder.classes_)
        num_sentiments = len(sentiment_encoder.classes_)
        # num_sentiments = 3
        # num_genres = 7

        text_model = TextClassifier(num_genres, num_sentiments).to(device)
        # audio_model = AudioClassifier(num_genres).to(device)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        text_model.load_state_dict(torch.load('best_text_model.pth'))
        # audio_model.load_state_dict(torch.load('best_audio_model.pth'))

        return text_model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


# Initialize models and tokenizer
try:
    logger.info("Initializing models and tokenizer...")
    text_model = load_text_model('best_text_model.pth')
    text_model.to(device)
    text_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("Models and tokenizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise


def analyze_lyrics(lyrics):
    logger.info("Starting lyrics analysis")
    logger.debug(f"Input lyrics: {lyrics[:100]}...")  # Log first 100 chars

    # Tokenize
    logger.debug("Tokenizing input text...")
    encoded = tokenizer.encode_plus(
        lyrics,
        add_special_tokens=True,
        max_length=MAX_LENGTH_TEXT,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    logger.debug(f"Encoded shape - Input IDs: {encoded['input_ids'].shape}, "
                 f"Attention Mask: {encoded['attention_mask'].shape}")

    # Move tensors to device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Make prediction
    logger.debug("Making prediction...")
    with torch.no_grad():
        outputs = text_model(input_ids, attention_mask)
        logger.debug(f"Raw outputs: {outputs}")
        predictions = torch.sigmoid(outputs)
        logger.debug(f"Predictions after sigmoid: {predictions}")
        sentiment_index = (predictions > 0.5).int().item()
        logger.debug(f"Final sentiment index: {sentiment_index}")

    return sentiment_index


@app.route('/api/analyze-lyrics', methods=['POST'])
def analyze_lyrics_route():
    logger.info("Lyrics analysis endpoint accessed")

    data = request.json
    if not data or 'lyrics' not in data:
        logger.warning("No lyrics data in request")
        return jsonify({"error": "No lyrics data"})

    lyrics = data['lyrics']
    logger.info(f"Lyrics: {lyrics[:100]}...")
    try:
        result = predict_complete_lyrics(lyrics)
        logger.info(f"Analysis complete. Sentiment: {result['text_analysis']}")
        beat = result['text_analysis']['genre']
        sentiment = 'Positive' if result['text_analysis']['sentiment'] == 2 else 'Negative'
        logger.info(f"Analysis complete. Sentiment: {sentiment}")
    except Exception as e:
        logger.error(f"Error during lyrics analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"})

    return jsonify({"sentiment": sentiment})


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Music Sentiment Analyzer API!"})


@app.route('/model-performance', methods=['GET'])
def model_performance():
    return render_template('performance.html')  # Ensure 'performance.html' exists

def analyze_audio(file_path):
    logger.info(f"Starting audio analysis for file: {file_path}")

    try:
        # Load audio file
        logger.debug("Loading audio file...")
        y_audio, sr = librosa.load(file_path, sr=22050)

        # Extract MFCCs
        logger.debug("Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=MFCC_N)

        # Pad or trim the MFCCs to the required width
        logger.debug(f"Original MFCC shape: {mfccs.shape}")
        if mfccs.shape[1] < TARGET_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, TARGET_LENGTH - mfccs.shape[1])),
                           mode='constant')
        else:
            mfccs = mfccs[:, :TARGET_LENGTH]
        logger.debug(f"Processed MFCC shape: {mfccs.shape}")

        # Correct reshaping for CNN input [batch_size, channels, height, width]
        mfccs_tensor = torch.FloatTensor(mfccs)
        mfccs_tensor = mfccs_tensor.unsqueeze(0)  # Add batch dimension
        mfccs_tensor = mfccs_tensor.unsqueeze(0)  # Add channel dimension
        logger.debug(f"Input tensor shape: {mfccs_tensor.shape}")

        mfccs_tensor = mfccs_tensor.to(device)

        # Load models and encoders if not already loaded
        genre_encoder, sentiment_encoder = load_encoders()
        num_genres = len(genre_encoder.classes_)

        audio_model = AudioClassifier(num_genres).to(device)
        audio_model.load_state_dict(torch.load('best_audio_model.pth'))
        audio_model.eval()

        # Make prediction
        logger.debug("Making prediction...")
        with torch.no_grad():
            outputs = audio_model(mfccs_tensor)
            predictions = torch.softmax(outputs, dim=1)
            predicted_genre_idx = torch.argmax(predictions, dim=1).item()

            # Get confidence scores
            confidence_scores = predictions[0].cpu().numpy()

            # Get predicted genre label
            predicted_genre = genre_encoder.inverse_transform([predicted_genre_idx])[0]

            # Get top 3 predictions with their probabilities
            top_3_indices = np.argsort(confidence_scores)[-3:][::-1]
            top_3_genres = genre_encoder.inverse_transform(top_3_indices)
            top_3_probabilities = confidence_scores[top_3_indices]

            # Format results
            top_3_predictions = [
                {"genre": genre, "probability": float(prob)}
                for genre, prob in zip(top_3_genres, top_3_probabilities)
            ]

            results = {
                "predicted_genre": predicted_genre,
                "confidence": float(confidence_scores[predicted_genre_idx]),
                "top_3_predictions": top_3_predictions
            }

            logger.info(f"Audio analysis complete. Predicted genre: {predicted_genre}")
            return results

    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}", exc_info=True)
        raise

def predict_genre_from_audio(audio_file, model, encoder, device):
    """
    Helper function to predict genre from audio file
    """
    try:
        # Load and preprocess audio
        y_audio, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=MFCC_N)

        # Pad or trim
        if mfccs.shape[1] < TARGET_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, TARGET_LENGTH - mfccs.shape[1])),
                           mode='constant')
        else:
            mfccs = mfccs[:, :TARGET_LENGTH]

            # Correct reshaping for CNN input
        mfccs_tensor = torch.FloatTensor(mfccs)
        mfccs_tensor = mfccs_tensor.unsqueeze(0)  # Add batch dimension
        mfccs_tensor = mfccs_tensor.unsqueeze(0)  # Add channel dimension
        mfccs_tensor = mfccs_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(mfccs_tensor)
            predictions = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(predictions, dim=1).item()

            # Get confidence scores
            confidence_scores = predictions[0].cpu().numpy()

            # Get top 3 predictions
            top_3_indices = np.argsort(confidence_scores)[-3:][::-1]
            top_3_genres = encoder.inverse_transform(top_3_indices)
            top_3_probabilities = confidence_scores[top_3_indices]

            return {
                "predicted_genre": encoder.inverse_transform([predicted_idx])[0],
                "confidence": float(confidence_scores[predicted_idx]),
                "top_3_predictions": [
                    {"genre": genre, "probability": float(prob)}
                    for genre, prob in zip(top_3_genres, top_3_probabilities)
                ]
            }

    except Exception as e:
        logger.error(f"Error in genre prediction: {str(e)}", exc_info=True)
        raise

    # Update the API endpoint accordingly


@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio_route():
    logger.info("Audio analysis endpoint accessed")

    try:
        if 'audio' not in request.files:
            logger.warning("No audio file in request")
            return jsonify({"error": "No audio file"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.warning("No selected audio file")
            return jsonify({"error": "No selected file"}), 400

            # Create temporary file to store the audio
        temp_path = "temp_audio.mp3"
        audio_file.save(temp_path)

        try:
            results = analyze_audio(temp_path)
            return jsonify(results)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return jsonify({"error": "Error processing audio file"}), 500


# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)