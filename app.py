"""
Flask Web Application for Tweet Sentiment Analysis
Provides a modern web interface for real-time sentiment prediction
"""

from flask import Flask, render_template, request, jsonify
import torch
import pickle
import logging
import os
import traceback
from pathlib import Path
import time
from datetime import datetime

# Import our existing model and preprocessing
from sentiment_model import SentimentLSTM
from data_preparation import AdvancedTweetPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Web-optimized sentiment predictor class"""
    
    def __init__(self, model_path="models/best_model.pth", vocab_path="models/vocabulary.pkl", device='cpu'):
        """Initialize the predictor with trained model and vocabulary"""
        self.device = device
        self.model = None
        self.word_to_idx = None
        self.idx_to_vocab = None
        self.preprocessor = AdvancedTweetPreprocessor()
        self.model_info = {}
        
        try:
            self.load_model(model_path, vocab_path)
            logger.info(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_model(self, model_path, vocab_path):
        """Load the trained model and vocabulary"""
        
        # Verify that files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        logger.info(f"📚 Loading vocabulary from {vocab_path}")
        
        # Load vocabulary
        try:
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                # Correction: use the correct keys from our vocabulary
                if 'vocab_to_idx' in vocab_data:
                    self.word_to_idx = vocab_data['vocab_to_idx']
                    self.idx_to_vocab = vocab_data['idx_to_vocab']
                else:
                    # Fallback for alternative format
                    self.word_to_idx = vocab_data
                    self.idx_to_vocab = {v: k for k, v in vocab_data.items()}
        except Exception as e:
            raise RuntimeError(f"Error loading vocabulary: {e}")
        
        vocab_size = len(self.word_to_idx)
        logger.info(f"📝 Vocabulary loaded: {vocab_size} words")
        
        # Model configuration (must match training)
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'use_attention': True
        }
        
        logger.info(f"🧠 Creating model with config: {model_config}")
        self.model = SentimentLSTM(**model_config)
        
        # Load state dict
        logger.info(f"📥 Loading model weights from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # Save model information
                self.model_info = {
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'val_acc': checkpoint.get('val_acc', 'Unknown'),
                    'val_f1': checkpoint.get('val_f1', 'Unknown'),
                    'training_time': checkpoint.get('training_time', 'Unknown')
                }
                logger.info(f"📊 Model info: Epoch {self.model_info['epoch']}, "
                           f"Val Acc: {self.model_info['val_acc']}, "
                           f"Val F1: {self.model_info['val_f1']}")
            else:
                self.model.load_state_dict(checkpoint)
                self.model_info = {'source': 'Direct state dict'}
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Test del modello con un input di prova
        try:
            test_input = torch.zeros((1, 100), dtype=torch.long).to(self.device)
            with torch.no_grad():
                test_output, _ = self.model(test_input)
                logger.info(f"🧪 Model test successful, output shape: {test_output.shape}")
        except Exception as e:
            logger.warning(f"⚠️ Model test failed: {e}")
    
    def text_to_tensor(self, text, max_length=100):
        """Convert text to tensor for model input"""
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text.strip():
            # Se il testo pulito è vuoto, usa il testo originale
            cleaned_text = text.lower().strip()
        
        # Tokenize
        tokens = cleaned_text.split()
        
        # Convert to indices
        unk_token = self.word_to_idx.get('<UNK>', 1)
        pad_token = self.word_to_idx.get('<PAD>', 0)
        
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(unk_token)
        
        # Pad or truncate
        if len(indices) < max_length:
            indices.extend([pad_token] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        return torch.tensor([indices], dtype=torch.long).to(self.device), cleaned_text
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                return {
                    'error': 'Empty text provided',
                    'sentiment': 'Unknown',
                    'confidence': 0
                }
            
            # Convert text to tensor
            input_tensor, cleaned_text = self.text_to_tensor(text)
            
            # Make prediction
            with torch.no_grad():
                outputs, attention_weights = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # CORRECTION: Correct mapping for our model
            # From our tests we know that: 0=Negative, 1=Positive
            sentiment = "Positive" if predicted_class == 1 else "Negative"
            
            # Calculate probabilities for both classes
            prob_negative = probabilities[0][0].item()  # Class 0 = Negative
            prob_positive = probabilities[0][1].item()  # Class 1 = Positive
            
            prediction_time = time.time() - start_time
            
            result = {
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'cleaned_text': cleaned_text,
                'probabilities': {
                    'positive': round(prob_positive * 100, 2),
                    'negative': round(prob_negative * 100, 2)
                },
                'prediction_time_ms': round(prediction_time * 1000, 2),
                'model_info': {
                    'predicted_class': predicted_class,
                    'device': str(self.device),
                    'tokens_processed': len(cleaned_text.split())
                }
            }
            
            logger.info(f"✅ Prediction: '{text[:50]}...' -> {sentiment} ({confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'error': f'Errore di predizione: {str(e)}',
                'sentiment': 'Unknown',
                'confidence': 0,
                'cleaned_text': text
            }

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tweet-sentiment-analysis-2024'
app.config['JSON_AS_ASCII'] = False  # Per supportare caratteri italiani

# Initialize predictor (global variable)
predictor = None

def init_predictor():
    """Initialize the predictor with error handling"""
    global predictor
    try:
        logger.info("🔄 Initializing sentiment predictor...")
        predictor = SentimentPredictor()
        return True
    except FileNotFoundError as e:
        logger.error(f"📁 Model files not found: {e}")
        logger.error("🔧 Please run training first: python train_sentiment_model.py")
        return False
    except Exception as e:
        logger.error(f"💥 Failed to initialize predictor: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        logger.warning("🔄 Predictor not initialized, attempting to initialize...")
        if not init_predictor():
            return jsonify({
                'error': 'Modello non disponibile. Assicurati che il modello sia stato addestrato.',
                'sentiment': 'Unknown',
                'confidence': 0,
                'suggestion': 'Esegui: python train_sentiment_model.py'
            }), 503
    
    try:
        # Get tweet text from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nessun dato JSON fornito'}), 400
        
        if 'text' not in data:
            return jsonify({'error': 'Campo "text" mancante'}), 400
        
        tweet_text = data['text']
        if not isinstance(tweet_text, str):
            return jsonify({'error': 'Il campo "text" deve essere una stringa'}), 400
            
        tweet_text = tweet_text.strip()
        if not tweet_text:
            return jsonify({'error': 'Testo vuoto fornito'}), 400
        
        if len(tweet_text) > 500:  # Limite ragionevole
            return jsonify({'error': 'Testo troppo lungo (massimo 500 caratteri)'}), 400
        
        # Make prediction
        result = predictor.predict(tweet_text)
        
        # Add metadata
        result['original_text'] = tweet_text
        result['timestamp'] = time.time()
        result['api_info'] = {
            'model_type': 'LSTM with Attention',
            'device': str(predictor.device),
            'version': '1.0'
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"💥 Prediction endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Errore interno del server: {str(e)}',
            'sentiment': 'Unknown',
            'confidence': 0
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global predictor
    
    status = 'healthy' if predictor is not None else 'model_not_loaded'
    
    health_info = {
        'status': status,
        'model_loaded': predictor is not None,
        'timestamp': time.time()
    }
    
    if predictor:
        health_info.update({
            'device': str(predictor.device),
            'vocab_size': len(predictor.word_to_idx) if predictor.word_to_idx else 0,
            'model_info': predictor.model_info
        })
    
    return jsonify(health_info)

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Tweet Sentiment Analysis API',
        'version': '1.0',
        'description': 'API per l\'analisi del sentiment di tweet usando LSTM con Attention',
        'endpoints': {
            '/': 'Interfaccia web principale',
            '/predict': 'POST - Predizione sentiment',
            '/health': 'GET - Stato del servizio',
            '/api/info': 'GET - Informazioni API'
        },
        'model': {
            'type': 'LSTM with Attention Mechanism',
            'architecture': 'Bidirectional LSTM + Multiple Pooling',
            'classes': ['Negative', 'Positive'],
            'max_text_length': 280
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint non trovato'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({'error': 'Metodo non consentito'}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"💥 Internal server error: {error}")
    return jsonify({'error': 'Errore interno del server'}), 500

if __name__ == '__main__':
    # Initialize predictor on startup
    logger.info("🚀 Starting Tweet Sentiment Analysis Web App")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🔍 Checking for model files...")
    
    # Verifica file del modello
    model_path = "models/best_model.pth"
    vocab_path = "models/vocabulary.pkl"
    
    if os.path.exists(model_path):
        logger.info(f"✅ Model file found: {model_path}")
    else:
        logger.warning(f"⚠️ Model file not found: {model_path}")
    
    if os.path.exists(vocab_path):
        logger.info(f"✅ Vocabulary file found: {vocab_path}")
    else:
        logger.warning(f"⚠️ Vocabulary file not found: {vocab_path}")
    
    if init_predictor():
        logger.info("✅ Predictor initialized successfully")
        logger.info("🌐 Web app ready! Access at: http://localhost:5000")
    else:
        logger.warning("⚠️ Predictor initialization failed")
        logger.warning("📝 Web app will start but predictions will not work")
        logger.warning("🔧 Please run: python train_sentiment_model.py")
    
    # Run the app
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("👋 App stopped by user")
    except Exception as e:
        logger.error(f"💥 App crashed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}") 