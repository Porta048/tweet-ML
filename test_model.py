import torch
import pandas as pd
from data_preparation import TweetPreprocessor
from sentiment_model import SentimentLSTM, SentimentTrainer, load_vocabulary
import os
import sys
import logging
import psutil
import warnings
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import time
import gc

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EdgeCaseHandler:
    """Gestore centralizzato per tutti i casi edge nel testing"""
    
    @staticmethod
    def validate_text(text: any) -> Tuple[bool, str, str]:
        """
        Valida e pulisce l'input testuale
        Returns: (is_valid, cleaned_text, warning_message)
        """
        warning = ""
        
        # Caso None
        if text is None:
            return False, "", "Input None rilevato"
        
        # Converti a stringa se necessario
        try:
            text_str = str(text)
        except Exception as e:
            return False, "", f"Impossibile convertire a stringa: {e}"
        
        # Caso stringa vuota
        if not text_str.strip():
            return False, "", "Testo vuoto dopo pulizia"
        
        # Caso troppo lungo (limite 1000 caratteri per sicurezza)
        if len(text_str) > 1000:
            text_str = text_str[:1000]
            warning = f"Testo troncato a 1000 caratteri (era {len(str(text))})"
        
        # Caso solo caratteri speciali/numeri
        clean_text = ''.join(c for c in text_str if c.isalpha() or c.isspace())
        if len(clean_text.strip()) < 3:
            warning += " Testo con poche lettere significative"
        
        return True, text_str, warning
    
    @staticmethod
    def validate_csv_file(file_path: str) -> Tuple[bool, str]:
        """Valida file CSV prima del caricamento"""
        if not os.path.exists(file_path):
            return False, f"File non trovato: {file_path}"
        
        if not file_path.lower().endswith(('.csv', '.tsv')):
            return False, "File deve essere CSV o TSV"
        
        # Controlla dimensione file (max 500MB)
        try:
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            if file_size > 500:
                return False, f"File troppo grande: {file_size:.1f}MB (max 500MB)"
        except Exception as e:
            return False, f"Errore controllo dimensione: {e}"
        
        return True, "File valido"
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Rileva automaticamente l'encoding del file"""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Leggi primi 1KB per test
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'  # Fallback
    
    @staticmethod
    def check_memory_usage() -> Dict[str, float]:
        """Controlla uso memoria corrente"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    @staticmethod
    def estimate_batch_memory(num_samples: int, max_length: int = 128) -> float:
        """Stima memoria necessaria per batch (in GB)"""
        # Stima approssimativa basata su dimensioni tensori
        bytes_per_sample = max_length * 4 * 8  # float32 * layers
        total_bytes = num_samples * bytes_per_sample
        return total_bytes / (1024**3)

class TweetSentimentPredictor:
    """Classe robusta per predizioni con gestione completa dei casi edge"""
    
    def __init__(self, model_path='models/best_model.pth', vocab_path='models/vocabulary.pkl'):
        self.device = None
        self.preprocessor = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.model = None
        self.trainer = None
        self.edge_handler = EdgeCaseHandler()
        
        # Inizializzazione con gestione errori completa
        if not self._initialize_components(model_path, vocab_path):
            raise RuntimeError("Inizializzazione fallita. Controlla i log per dettagli.")
    
    def _initialize_components(self, model_path: str, vocab_path: str) -> bool:
        """Inizializza tutti i componenti con gestione errori"""
        try:
            # 1. Controlla file esistenti
            if not os.path.exists(model_path):
                logger.error(f"❌ Modello non trovato: {model_path}")
                return False
            
            if not os.path.exists(vocab_path):
                logger.error(f"❌ Vocabolario non trovato: {vocab_path}")
                return False
            
            # 2. Controlla memoria disponibile
            memory_info = self.edge_handler.check_memory_usage()
            logger.info(f"💾 Memoria disponibile: {memory_info['available_gb']:.1f}GB")
            
            if memory_info['available_gb'] < 1.0:
                logger.warning("⚠️ Memoria disponibile < 1GB, potrebbero esserci problemi")
            
            # 3. Inizializza device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"🔧 Device: {self.device}")
            
            # 4. Inizializza preprocessor
            try:
                self.preprocessor = TweetPreprocessor()
                logger.info("✅ Preprocessor inizializzato")
            except Exception as e:
                logger.error(f"❌ Errore inizializzazione preprocessor: {e}")
                return False
            
            # 5. Carica vocabolario
            try:
                logger.info("📚 Caricamento vocabolario...")
                self.vocab_to_idx, self.idx_to_vocab = load_vocabulary(vocab_path)
                vocab_size = len(self.vocab_to_idx)
                
                if vocab_size < 100:
                    logger.warning(f"⚠️ Vocabolario molto piccolo: {vocab_size} parole")
                
                logger.info(f"✅ Vocabolario caricato: {vocab_size} parole")
            except Exception as e:
                logger.error(f"❌ Errore caricamento vocabolario: {e}")
                return False
            
            # 6. Carica modello
            try:
                logger.info("🧠 Caricamento modello...")
                
                # Carica checkpoint per verificare architettura
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' not in checkpoint:
                    logger.error("❌ Checkpoint corrotto: manca model_state_dict")
                    return False
                
                # Determina parametri architettura da checkpoint
                state_dict = checkpoint['model_state_dict']
                
                # Estrai parametri dal state dict
                embedding_dim = state_dict['embedding.weight'].shape[1]
                hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM ha 4 gates
                
                # Determina se ha attention
                has_attention = any('attention' in key for key in state_dict.keys())
                
                # Conta layers LSTM
                lstm_layers = max([int(key.split('_l')[1].split('_')[0]) for key in state_dict.keys() 
                                 if 'lstm.weight_ih_l' in key]) + 1
                
                logger.info(f"📋 Architettura rilevata:")
                logger.info(f"   - Embedding: {embedding_dim}")
                logger.info(f"   - Hidden: {hidden_dim}")
                logger.info(f"   - LSTM Layers: {lstm_layers}")
                logger.info(f"   - Attention: {has_attention}")
                
                # Crea modello con architettura corretta
                self.model = SentimentLSTM(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_layers=lstm_layers,
                    dropout=0.3,
                    use_attention=has_attention
                )
                
                # Carica pesi
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Verifica compatibilità
                val_acc = checkpoint.get('val_acc', 'N/A')
                val_f1 = checkpoint.get('val_f1', 'N/A')
                
                logger.info(f"✅ Modello caricato")
                logger.info(f"   - Accuratezza validazione: {val_acc}")
                logger.info(f"   - F1-score validazione: {val_f1}")
                
            except Exception as e:
                logger.error(f"❌ Errore caricamento modello: {e}")
                return False
            
            # 7. Inizializza trainer
            try:
                self.trainer = SentimentTrainer(self.model, self.device)
                logger.info("✅ Trainer inizializzato")
            except Exception as e:
                logger.error(f"❌ Errore inizializzazione trainer: {e}")
                return False
            
            logger.info("🎉 Inizializzazione completata con successo!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore critico durante inizializzazione: {e}")
            return False
    
    def predict_single(self, tweet_text: any) -> Tuple[str, float, str, List[str]]:
        """
        Predice il sentiment di un singolo tweet con gestione robusta
        Returns: (sentiment, confidence, clean_text, warnings)
        """
        warnings_list = []
        
        try:
            # Valida input
            is_valid, cleaned_text, warning = self.edge_handler.validate_text(tweet_text)
            
            if warning:
                warnings_list.append(warning)
            
            if not is_valid:
                return "Neutrale", 0.5, cleaned_text, warnings_list
            
            # Pulisci il testo
            try:
                clean_text = self.preprocessor.clean_text(cleaned_text)
            except Exception as e:
                warnings_list.append(f"Errore preprocessing: {e}")
                return "Neutrale", 0.5, str(tweet_text), warnings_list
            
            if not clean_text.strip():
                warnings_list.append("Testo vuoto dopo preprocessing")
                return "Neutrale", 0.5, clean_text, warnings_list
            
            # Fai la predizione
            try:
                sentiment, confidence = self.trainer.predict(clean_text, self.vocab_to_idx)
                
                # Validazione output
                if not isinstance(sentiment, str) or sentiment not in ["Positivo", "Negativo"]:
                    warnings_list.append(f"Sentiment non valido: {sentiment}")
                    sentiment = "Neutrale"
                    confidence = 0.5
                
                if not (0 <= confidence <= 1):
                    warnings_list.append(f"Confidenza non valida: {confidence}")
                    confidence = max(0, min(1, confidence))
                
            except Exception as e:
                warnings_list.append(f"Errore predizione: {e}")
                return "Neutrale", 0.5, clean_text, warnings_list
            
            return sentiment, confidence, clean_text, warnings_list
            
        except Exception as e:
            logger.error(f"Errore critico in predict_single: {e}")
            return "Neutrale", 0.5, str(tweet_text), [f"Errore critico: {e}"]
    
    def predict_batch(self, tweet_list: List[any], batch_size: int = 32, 
                     show_progress: bool = True) -> pd.DataFrame:
        """Predice il sentiment per una lista di tweet con gestione memoria"""
        
        if not tweet_list:
            logger.warning("Lista tweet vuota")
            return pd.DataFrame(columns=['index', 'original_text', 'clean_text', 
                                       'sentiment', 'confidence', 'warnings'])
        
        logger.info(f"🔄 Analisi batch di {len(tweet_list)} tweet...")
        
        # Controlla memoria disponibile
        memory_info = self.edge_handler.check_memory_usage()
        estimated_memory = self.edge_handler.estimate_batch_memory(len(tweet_list))
        
        if estimated_memory > memory_info['available_gb'] * 0.8:
            logger.warning(f"⚠️ Memoria potenzialmente insufficiente")
            logger.warning(f"   Stimata: {estimated_memory:.2f}GB")
            logger.warning(f"   Disponibile: {memory_info['available_gb']:.2f}GB")
            batch_size = max(1, batch_size // 2)
            logger.info(f"   Ridotto batch_size a {batch_size}")
        
        results = []
        errors_count = 0
        
        # Processa in batch per gestire memoria
        iterator = range(0, len(tweet_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(tweet_list))
            batch = tweet_list[start_idx:end_idx]
            
            # Processa batch corrente
            for i, tweet in enumerate(batch):
                global_idx = start_idx + i
                
                try:
                    sentiment, confidence, clean_text, warnings = self.predict_single(tweet)
                    
                    if warnings:
                        errors_count += 1
                    
                    results.append({
                        'index': global_idx,
                        'original_text': str(tweet) if tweet is not None else '',
                        'clean_text': clean_text,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'warnings': '; '.join(warnings) if warnings else ''
                    })
                    
                except Exception as e:
                    logger.error(f"Errore elaborazione tweet {global_idx}: {e}")
                    errors_count += 1
                    
                    results.append({
                        'index': global_idx,
                        'original_text': str(tweet) if tweet is not None else '',
                        'clean_text': '',
                        'sentiment': 'Neutrale',
                        'confidence': 0.5,
                        'warnings': f'Errore elaborazione: {e}'
                    })
            
            # Cleanup memoria periodico
            if start_idx % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f"✅ Batch completato: {len(results)} tweet processati")
        if errors_count > 0:
            logger.warning(f"⚠️ {errors_count} tweet con warnings/errori")
        
        return pd.DataFrame(results)
    
    def analyze_csv(self, csv_path: str, text_column: str = 'text', 
                   output_path: Optional[str] = None, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Analizza tweet da CSV con gestione robusta"""
        
        logger.info(f"📄 Analisi CSV: {csv_path}")
        
        # Valida file
        is_valid, error_msg = self.edge_handler.validate_csv_file(csv_path)
        if not is_valid:
            logger.error(f"❌ {error_msg}")
            return None
        
        # Rileva encoding
        encoding = self.edge_handler.detect_encoding(csv_path)
        logger.info(f"🔤 Encoding rilevato: {encoding}")
        
        try:
            # Carica CSV con gestione errori
            read_params = {
                'encoding': encoding,
                'on_bad_lines': 'skip',  # Salta righe problematiche
                'engine': 'python'  # Engine più robusto
            }
            
            if max_rows:
                read_params['nrows'] = max_rows
                logger.info(f"📊 Limitato a {max_rows} righe")
            
            df = pd.read_csv(csv_path, **read_params)
            logger.info(f"✅ CSV caricato: {len(df)} righe, {len(df.columns)} colonne")
            
        except Exception as e:
            logger.error(f"❌ Errore lettura CSV: {e}")
            
            # Tentativo con encoding alternativo
            try:
                logger.info("🔄 Tentativo con encoding utf-8...")
                df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
                logger.info(f"✅ CSV caricato con utf-8: {len(df)} righe")
            except Exception as e2:
                logger.error(f"❌ Fallito anche con utf-8: {e2}")
                return None
        
        # Verifica colonna testo
        if text_column not in df.columns:
            logger.error(f"❌ Colonna '{text_column}' non trovata")
            logger.info(f"Colonne disponibili: {list(df.columns)}")
            return None
        
        # Rimuovi righe con testo vuoto/None
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.strip() != '']
        
        if len(df) < initial_count:
            logger.warning(f"⚠️ Rimosse {initial_count - len(df)} righe vuote/None")
        
        if len(df) == 0:
            logger.error("❌ Nessuna riga valida rimasta")
            return None
        
        logger.info(f"🔄 Analisi di {len(df)} tweet validi...")
        
        # Processa tweet
        try:
            tweet_list = df[text_column].tolist()
            results_df = self.predict_batch(tweet_list, show_progress=True)
            
            # Unisci risultati con DataFrame originale
            df = df.reset_index(drop=True)
            df['predicted_sentiment'] = results_df['sentiment']
            df['confidence'] = results_df['confidence']
            df['clean_text'] = results_df['clean_text']
            df['warnings'] = results_df['warnings']
            
        except Exception as e:
            logger.error(f"❌ Errore elaborazione: {e}")
            return None
        
        # Salva risultati
        if output_path:
            try:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"💾 Risultati salvati: {output_path}")
            except Exception as e:
                logger.error(f"❌ Errore salvataggio: {e}")
        
        # Statistiche finali
        try:
            sentiment_counts = df['predicted_sentiment'].value_counts()
            warnings_count = (df['warnings'] != '').sum()
            
            logger.info("\n" + "="*50)
            logger.info("📊 RISULTATI ANALISI CSV")
            logger.info("="*50)
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                emoji = "😊" if sentiment == "Positivo" else "😞" if sentiment == "Negativo" else "😐"
                logger.info(f"{emoji} {sentiment}: {count} tweet ({percentage:.1f}%)")
            
            avg_confidence = df['confidence'].mean()
            logger.info(f"\n🎯 Confidenza media: {avg_confidence:.4f}")
            
            if warnings_count > 0:
                logger.warning(f"⚠️ Tweet con warnings: {warnings_count} ({warnings_count/len(df)*100:.1f}%)")
            
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"❌ Errore calcolo statistiche: {e}")
        
        return df

def test_edge_cases():
    """Testa specificamente i casi edge"""
    print("\n" + "="*60)
    print("🧪 TEST SPECIFICI PER CASI EDGE")
    print("="*60)
    
    try:
        predictor = TweetSentimentPredictor()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return
    
    # Test cases problematici
    edge_cases = [
        None,                                    # None
        "",                                      # Stringa vuota
        "   ",                                   # Solo spazi
        "123456789",                            # Solo numeri
        "!@#$%^&*()",                          # Solo simboli
        "a",                                    # Troppo corto
        "😀😁😂🤣😃😄😅😆😉😊",                    # Solo emoji
        "http://example.com",                   # Solo URL
        "@user #hashtag",                       # Solo menzioni/hashtag
        "a" * 1500,                           # Troppo lungo
        "Great day! " * 100,                   # Ripetitivo
        "Hello\nWorld\t\r",                    # Caratteri speciali
        "Текст на русском",                     # Cirillico
        "نص عربي",                              # Arabo
        "🔥🔥🔥 AMAZING!!! 🔥🔥🔥",                # Misto emoji/testo
        12345,                                  # Numero
        ["not", "a", "string"],               # Lista
        {"text": "dict"},                      # Dizionario
        b"bytes text",                         # Bytes
    ]
    
    print(f"📋 Testing {len(edge_cases)} casi edge...")
    print()
    
    for i, case in enumerate(edge_cases, 1):
        try:
            sentiment, confidence, clean_text, warnings = predictor.predict_single(case)
            
            print(f"Test {i:2d}: {type(case).__name__}")
            print(f"   Input: {repr(case)}")
            print(f"   Clean: {repr(clean_text)}")
            print(f"   Result: {sentiment} ({confidence:.3f})")
            if warnings:
                print(f"   ⚠️ Warnings: {'; '.join(warnings)}")
            print()
            
        except Exception as e:
            print(f"Test {i:2d}: ❌ ERRORE - {e}")
            print()
    
    print("✅ Test casi edge completato!")

def interactive_test():
    """Modalità interattiva migliorata"""
    print("\n" + "="*60)
    print("💬 MODALITÀ INTERATTIVA - SENTIMENT ANALYSIS")
    print("="*60)
    print("Comandi speciali:")
    print("  'quit' - Esci")
    print("  'test' - Lancia test casi edge")
    print("  'memory' - Mostra info memoria")
    print("  'stats' - Mostra statistiche sessione")
    print("-" * 60)
    
    try:
        predictor = TweetSentimentPredictor()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return
    
    # Statistiche sessione
    session_stats = {
        'total_predictions': 0,
        'positive': 0,
        'negative': 0,
        'errors': 0,
        'avg_confidence': 0,
        'confidences': []
    }
    
    while True:
        try:
            tweet = input("\n📝 Tweet da analizzare: ").strip()
            
            if tweet.lower() == 'quit':
                break
            elif tweet.lower() == 'test':
                test_edge_cases()
                continue
            elif tweet.lower() == 'memory':
                memory = EdgeCaseHandler.check_memory_usage()
                print(f"\n💾 Memoria sistema:")
                print(f"   Totale: {memory['total_gb']:.1f}GB")
                print(f"   Disponibile: {memory['available_gb']:.1f}GB")
                print(f"   Usata: {memory['used_percent']:.1f}%")
                continue
            elif tweet.lower() == 'stats':
                print(f"\n📊 Statistiche sessione:")
                print(f"   Predizioni totali: {session_stats['total_predictions']}")
                if session_stats['total_predictions'] > 0:
                    print(f"   Positivi: {session_stats['positive']} ({session_stats['positive']/session_stats['total_predictions']*100:.1f}%)")
                    print(f"   Negativi: {session_stats['negative']} ({session_stats['negative']/session_stats['total_predictions']*100:.1f}%)")
                    print(f"   Errori: {session_stats['errors']}")
                    if session_stats['confidences']:
                        print(f"   Confidenza media: {np.mean(session_stats['confidences']):.3f}")
                continue
            
            if not tweet:
                print("⚠️ Inserisci un testo valido")
                continue
            
            # Predizione
            sentiment, confidence, clean_text, warnings = predictor.predict_single(tweet)
            
            # Aggiorna statistiche
            session_stats['total_predictions'] += 1
            session_stats['confidences'].append(confidence)
            
            if sentiment == "Positivo":
                session_stats['positive'] += 1
            elif sentiment == "Negativo":
                session_stats['negative'] += 1
            
            if warnings:
                session_stats['errors'] += 1
            
            # Output risultati
            print("\n" + "="*50)
            print(f"📄 Testo originale: {tweet}")
            if clean_text != tweet:
                print(f"🧹 Testo pulito: {clean_text}")
            
            # Emoji per sentiment
            emoji = "😊" if sentiment == "Positivo" else "😞" if sentiment == "Negativo" else "😐"
            print(f"{emoji} Sentiment: {sentiment}")
            print(f"🎯 Confidenza: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Livello confidenza
            if confidence > 0.8:
                conf_level = "🟢 Alta"
            elif confidence > 0.6:
                conf_level = "🟡 Media"
            else:
                conf_level = "🔴 Bassa"
            print(f"📊 Livello: {conf_level}")
            
            # Warnings
            if warnings:
                print(f"⚠️ Avvisi: {'; '.join(warnings)}")
            
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Uscita forzata!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            session_stats['errors'] += 1
    
    # Statistiche finali
    if session_stats['total_predictions'] > 0:
        print(f"\n📊 Statistiche finali sessione:")
        print(f"   Total predizioni: {session_stats['total_predictions']}")
        print(f"   Accuratezza stimata: {((session_stats['total_predictions'] - session_stats['errors'])/session_stats['total_predictions']*100):.1f}%")
    
    print("\n👋 Arrivederci!")

def test_examples():
    """Test su esempi predefiniti migliorato"""
    print("\n" + "="*60)
    print("📋 TEST SU ESEMPI PREDEFINITI")
    print("="*60)
    
    try:
        predictor = TweetSentimentPredictor()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return
    
    # Esempi diversificati
    test_tweets = [
        # Positivi chiari
        "I absolutely love this new restaurant! The food was amazing! 😍",
        "Just got promoted at work! So excited and grateful! 🎉",
        "Beautiful sunset tonight. Nature is incredible! 🌅",
        "@user thanks for the amazing service! Highly recommend!",
        "#MondayMotivation feeling positive about this week ahead!",
        "Just finished reading an incredible book. Mind blown! 📚",
        
        # Negativi chiari  
        "Worst customer service ever. Very disappointed and frustrated.",
        "This movie was terrible. Waste of time and money.",
        "Traffic is horrible today. Going to be late again...",
        "My phone broke again. This is so annoying!",
        "Can't believe how rude people can be sometimes. Very sad.",
        "Another day, another disappointment. Life is hard.",
        
        # Casi ambigui/difficili
        "The weather is okay I guess. Nothing special.",
        "Having a great time at the beach today! Perfect weather! ☀️",
        "Loved spending time with family this weekend. Very happy!",
        
        # Casi edge
        "😊😊😊",  # Solo emoji
        "AMAZING!!! 🔥🔥🔥",  # Caps + emoji
        "Not bad, could be better though",  # Ambiguo
        "Thanks... I guess",  # Sarcastico
    ]
    
    print(f"🔄 Analisi di {len(test_tweets)} tweet di esempio...")
    
    try:
        results = predictor.predict_batch(test_tweets, show_progress=True)
        
        # Mostra risultati dettagliati
        print(f"\n📊 RISULTATI DETTAGLIATI")
        print("="*80)
        
        for _, row in results.iterrows():
            idx = row['index'] + 1
            original = row['original_text']
            sentiment = row['sentiment']
            confidence = row['confidence']
            warnings = row['warnings']
            
            emoji = "😊" if sentiment == "Positivo" else "😞" if sentiment == "Negativo" else "😐"
            
            print(f"\n{idx:2d}. {original}")
            print(f"    {emoji} {sentiment} (Confidenza: {confidence:.3f})")
            
            if warnings:
                print(f"    ⚠️ {warnings}")
        
        # Statistiche aggregate
        sentiment_counts = results['sentiment'].value_counts()
        warnings_count = (results['warnings'] != '').sum()
        
        print(f"\n" + "="*50)
        print(f"📈 STATISTICHE FINALI")
        print("="*50)
        
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results)) * 100
            emoji = "😊" if sentiment == "Positivo" else "😞" if sentiment == "Negativo" else "😐"
            print(f"{emoji} {sentiment}: {count}/{len(results)} ({percentage:.1f}%)")
        
        avg_confidence = results['confidence'].mean()
        print(f"\n🎯 Confidenza media: {avg_confidence:.3f}")
        
        if warnings_count > 0:
            print(f"⚠️ Tweet con warnings: {warnings_count}/{len(results)} ({warnings_count/len(results)*100:.1f}%)")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Errore durante test esempi: {e}")
        print(f"❌ Errore durante test: {e}")

def main():
    """Funzione principale migliorata"""
    print("🤖 TWEET SENTIMENT ANALYSIS - TEST MODELLO")
    print("="*70)
    
    # Controlla prerequisiti
    if not os.path.exists('models/best_model.pth'):
        print("❌ Modello non trovato!")
        print("💡 Prima esegui: python train_sentiment_model.py")
        return
    
    if not os.path.exists('models/vocabulary.pkl'):
        print("❌ Vocabolario non trovato!")
        print("💡 Prima esegui: python train_sentiment_model.py")
        return
    
    # Controlla memoria
    memory_info = EdgeCaseHandler.check_memory_usage()
    print(f"💾 Memoria disponibile: {memory_info['available_gb']:.1f}GB")
    
    if memory_info['available_gb'] < 1.0:
        print("⚠️ Attenzione: memoria disponibile bassa!")
    
    print("\n🎯 Seleziona modalità:")
    print("1. 📋 Test esempi predefiniti")
    print("2. 💬 Modalità interattiva")
    print("3. 📄 Analizza file CSV")
    print("4. 🧪 Test casi edge")
    print("5. 📊 Info sistema")
    
    try:
        choice = input("\n🔢 Scelta (1-5): ").strip()
        
        if choice == '1':
            test_examples()
        elif choice == '2':
            interactive_test()
        elif choice == '3':
            csv_path = input("📂 Percorso file CSV: ").strip()
            if not csv_path:
                print("❌ Percorso non valido")
                return
                
            if os.path.exists(csv_path):
                text_column = input("📝 Colonna testo (default: 'text'): ").strip() or 'text'
                output_path = input("💾 File output (opzionale): ").strip() or None
                
                # Chiedi limite righe per file grandi
                try:
                    max_rows_input = input("📊 Max righe (opzionale): ").strip()
                    max_rows = int(max_rows_input) if max_rows_input else None
                except ValueError:
                    max_rows = None
                
                predictor = TweetSentimentPredictor()
                predictor.analyze_csv(csv_path, text_column, output_path, max_rows)
            else:
                print("❌ File CSV non trovato!")
        elif choice == '4':
            test_edge_cases()
        elif choice == '5':
            memory = EdgeCaseHandler.check_memory_usage()
            print(f"\n💻 INFORMAZIONI SISTEMA")
            print("="*40)
            print(f"💾 Memoria totale: {memory['total_gb']:.1f}GB")
            print(f"💾 Memoria disponibile: {memory['available_gb']:.1f}GB")
            print(f"💾 Memoria usata: {memory['used_percent']:.1f}%")
            print(f"🔧 GPU disponibile: {'Sì' if torch.cuda.is_available() else 'No'}")
            if torch.cuda.is_available():
                print(f"🔧 GPU: {torch.cuda.get_device_name()}")
            print(f"🐍 Python: {sys.version}")
            print(f"🔥 PyTorch: {torch.__version__}")
        else:
            print("❌ Scelta non valida!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Uscita forzata!")
    except Exception as e:
        logger.error(f"Errore in main: {e}")
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main() 