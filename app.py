"""
Flask Web Application for Tweet Sentiment Analysis
Provides a modern web interface for real-time sentiment prediction
"""

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import logging
import os
from pathlib import Path

# Import our existing model and preprocessing
from sentiment_model import SentimentLSTM
from data_preparation import TweetPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSentimentPredictor:
    """Web-optimized sentiment predictor class"""
    
    def __init__(self, model_path="models/best_model.pth", vocab_path="models/vocabulary.pkl"):
        """Initialize the predictor with trained model and vocabulary"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.word_to_idx = None
        self.idx_to_vocab = None
        self.preprocessor = TweetPreprocessor()
        
        try:
            self.load_model(model_path, vocab_path)
            logger.info(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def load_model(self, model_path, vocab_path):
        """Load the trained model and vocabulary"""
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.word_to_idx = vocab_data['vocab_to_idx']
            self.idx_to_vocab = vocab_data['idx_to_vocab']
        
        # Load model
        vocab_size = len(self.word_to_idx)
        
        # Model configuration (should match training config)
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'use_attention': True
        }
        
        self.model = SentimentLSTM(**model_config)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def text_to_tensor(self, text, max_length=100):
        """Convert text to tensor for model input"""
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Tokenize
        tokens = cleaned_text.lower().split()
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 0)) for token in tokens]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([self.word_to_idx.get('<PAD>', 0)] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return torch.tensor([indices], dtype=torch.long).to(self.device), cleaned_text
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        try:
            # Convert text to tensor
            input_tensor, cleaned_text = self.text_to_tensor(text)
            
            # Make prediction
            with torch.no_grad():
                outputs, attention_weights = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Map prediction to sentiment (0=Positive, 1=Negative)
            sentiment = "Positive" if predicted_class == 0 else "Negative"
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'cleaned_text': cleaned_text,
                'probabilities': {
                    'positive': round(probabilities[0][0].item() * 100, 2),
                    'negative': round(probabilities[0][1].item() * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'sentiment': 'Unknown',
                'confidence': 0,
                'cleaned_text': text
            }

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tweet-sentiment-analysis-2024'

# Initialize predictor (global variable)
predictor = None

def init_predictor():
    """Initialize the predictor with error handling"""
    global predictor
    try:
        predictor = WebSentimentPredictor()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return False

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """API endpoint for sentiment prediction"""
    global predictor
    
    # Check if predictor is initialized
    if predictor is None:
        if not init_predictor():
            return jsonify({
                'error': 'Model not available. Please ensure the model is trained.',
                'sentiment': 'Unknown',
                'confidence': 0
            }), 500
    
    try:
        # Get tweet text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        tweet_text = data['text'].strip()
        if not tweet_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = predictor.predict(tweet_text)
        
        # Add metadata
        result['original_text'] = tweet_text
        result['model_info'] = {
            'model_type': 'LSTM with Attention',
            'device': str(predictor.device)
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'sentiment': 'Unknown',
            'confidence': 0
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global predictor
    status = 'healthy' if predictor is not None else 'model_not_loaded'
    return jsonify({
        'status': status,
        'model_loaded': predictor is not None,
        'device': str(predictor.device) if predictor else 'unknown'
    })

if __name__ == '__main__':
    # Initialize predictor on startup
    logger.info("🚀 Starting Tweet Sentiment Analysis Web App")
    
    if init_predictor():
        logger.info("✅ Predictor initialized successfully")
    else:
        logger.warning("⚠️ Predictor initialization failed - web app will start but predictions may not work")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 