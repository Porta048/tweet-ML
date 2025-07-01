import torch
import pandas as pd
from data_preparation import TweetPreprocessor
from sentiment_model import SentimentLSTM, SentimentTrainer, load_vocabulary
import os

class TweetSentimentPredictor:
    """Classe per fare predizioni su nuovi tweet usando il modello addestrato"""
    
    def __init__(self, model_path='models/best_model.pth', vocab_path='models/vocabulary.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TweetPreprocessor()
        
        # Carica il vocabolario
        print("Caricamento vocabolario...")
        self.vocab_to_idx, self.idx_to_vocab = load_vocabulary(vocab_path)
        vocab_size = len(self.vocab_to_idx)
        
        # Crea il modello con la stessa architettura
        print("Caricamento modello...")
        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        )
        
        # Carica i pesi del modello
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Crea il trainer per usare il metodo predict
        self.trainer = SentimentTrainer(self.model, self.device)
        
        print(f"Modello caricato con accuratezza di validazione: {checkpoint['val_acc']:.4f}")
        print(f"Vocabolario: {vocab_size} parole")
        print(f"Device: {self.device}")
    
    def predict_single(self, tweet_text):
        """Predice il sentiment di un singolo tweet"""
        # Pulisci il testo
        clean_text = self.preprocessor.clean_text(tweet_text)
        
        if not clean_text.strip():
            return "Neutrale", 0.5, clean_text
        
        # Fai la predizione
        sentiment, confidence = self.trainer.predict(clean_text, self.vocab_to_idx)
        
        return sentiment, confidence, clean_text
    
    def predict_batch(self, tweet_list):
        """Predice il sentiment per una lista di tweet"""
        results = []
        
        for i, tweet in enumerate(tweet_list):
            sentiment, confidence, clean_text = self.predict_single(tweet)
            results.append({
                'index': i,
                'original_text': tweet,
                'clean_text': clean_text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return pd.DataFrame(results)
    
    def analyze_csv(self, csv_path, text_column='text', output_path=None):
        """Analizza tweet da un file CSV"""
        print(f"Caricamento file CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Colonna '{text_column}' non trovata nel CSV")
        
        print(f"Analisi di {len(df)} tweet...")
        
        # Applica la predizione a tutti i tweet
        results = []
        for idx, tweet in df[text_column].items():
            sentiment, confidence, clean_text = self.predict_single(str(tweet))
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'clean_text': clean_text
            })
        
        # Aggiungi i risultati al DataFrame originale
        results_df = pd.DataFrame(results)
        df['predicted_sentiment'] = results_df['sentiment']
        df['confidence'] = results_df['confidence']
        df['clean_text'] = results_df['clean_text']
        
        # Salva i risultati se specificato
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Risultati salvati in: {output_path}")
        
        # Statistiche
        sentiment_counts = df['predicted_sentiment'].value_counts()
        print("\nRisultati dell'analisi:")
        print("-" * 30)
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{sentiment}: {count} tweet ({percentage:.1f}%)")
        
        avg_confidence = df['confidence'].mean()
        print(f"\nConfidenza media: {avg_confidence:.4f}")
        
        return df

def interactive_test():
    """Modalità interattiva per testare singoli tweet"""
    print("\n" + "="*50)
    print("MODALITÀ INTERATTIVA - TEST SENTIMENT ANALYSIS")
    print("="*50)
    print("Inserisci 'quit' per uscire")
    print("-" * 50)
    
    # Carica il predictor
    try:
        predictor = TweetSentimentPredictor()
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        print("Assicurati di aver addestrato il modello prima!")
        return
    
    while True:
        # Input dell'utente
        tweet = input("\nInserisci il tweet da analizzare: ").strip()
        
        if tweet.lower() == 'quit':
            print("Arrivederci!")
            break
        
        if not tweet:
            print("Per favore inserisci un testo valido.")
            continue
        
        # Predizione
        sentiment, confidence, clean_text = predictor.predict_single(tweet)
        
        # Output
        print("\n" + "-" * 40)
        print(f"Testo originale: {tweet}")
        print(f"Testo pulito: {clean_text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidenza: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Interpretazione confidence
        if confidence > 0.8:
            conf_level = "Alta"
        elif confidence > 0.6:
            conf_level = "Media"
        else:
            conf_level = "Bassa"
        
        print(f"Livello di confidenza: {conf_level}")
        print("-" * 40)

def test_examples():
    """Testa il modello su esempi predefiniti"""
    print("\n" + "="*50)
    print("TEST SU ESEMPI PREDEFINITI")
    print("="*50)
    
    # Carica il predictor
    try:
        predictor = TweetSentimentPredictor()
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        print("Assicurati di aver addestrato il modello prima!")
        return
    
    # Esempi di tweet
    test_tweets = [
        "I absolutely love this new restaurant! The food was amazing! 😍",
        "Worst customer service ever. Very disappointed and frustrated.",
        "Having a great time at the beach today! Perfect weather! ☀️",
        "This movie was terrible. Waste of time and money.",
        "Just got promoted at work! So excited and grateful! 🎉",
        "Traffic is horrible today. Going to be late again...",
        "Beautiful sunset tonight. Nature is incredible! 🌅",
        "My phone broke again. This is so annoying!",
        "Loved spending time with family this weekend. Very happy!",
        "The weather is okay I guess. Nothing special.",
        "@user thanks for the amazing service! Highly recommend!",
        "#MondayMotivation feeling positive about this week ahead!",
        "Can't believe how rude people can be sometimes. Very sad.",
        "Just finished reading an incredible book. Mind blown! 📚",
        "Another day, another disappointment. Life is hard."
    ]
    
    print(f"Analisi di {len(test_tweets)} tweet di esempio...")
    print("\n")
    
    # Analizza ogni tweet
    results = predictor.predict_batch(test_tweets)
    
    # Mostra i risultati
    for _, row in results.iterrows():
        print(f"Tweet {row['index']+1}:")
        print(f"  Testo: {row['original_text']}")
        print(f"  Sentiment: {row['sentiment']} (Confidenza: {row['confidence']:.4f})")
        print()
    
    # Statistiche finali
    sentiment_counts = results['sentiment'].value_counts()
    print("\nRisultati finali:")
    print("-" * 20)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        print(f"{sentiment}: {count}/{len(results)} ({percentage:.1f}%)")
    
    avg_confidence = results['confidence'].mean()
    print(f"\nConfidenza media: {avg_confidence:.4f}")

def main():
    """Funzione principale"""
    print("TWEET SENTIMENT ANALYSIS - TEST MODELLO")
    print("="*60)
    
    # Controlla se il modello esiste
    if not os.path.exists('models/best_model.pth'):
        print("❌ Modello non trovato!")
        print("Per prima cosa esegui 'python train_sentiment_model.py' per addestrare il modello.")
        return
    
    if not os.path.exists('models/vocabulary.pkl'):
        print("❌ Vocabolario non trovato!")
        print("Per prima cosa esegui 'python train_sentiment_model.py' per addestrare il modello.")
        return
    
    print("✅ Modello e vocabolario trovati!")
    print("\nScegli una modalità:")
    print("1. Test su esempi predefiniti")
    print("2. Modalità interattiva")
    print("3. Analizza file CSV")
    
    choice = input("\nInserisci la tua scelta (1-3): ").strip()
    
    if choice == '1':
        test_examples()
    elif choice == '2':
        interactive_test()
    elif choice == '3':
        csv_path = input("Inserisci il percorso del file CSV: ").strip()
        if os.path.exists(csv_path):
            text_column = input("Nome della colonna con il testo (default: 'text'): ").strip() or 'text'
            output_path = input("Percorso file output (opzionale): ").strip() or None
            
            predictor = TweetSentimentPredictor()
            predictor.analyze_csv(csv_path, text_column, output_path)
        else:
            print("File CSV non trovato!")
    else:
        print("Scelta non valida!")

if __name__ == "__main__":
    main() 