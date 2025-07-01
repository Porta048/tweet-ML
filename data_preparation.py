import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import emoji
import os
from datasets import load_dataset

# Download delle risorse NLTK necessarie
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TweetPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Pulisce il testo del tweet rimuovendo elementi non necessari"""
        if pd.isna(text):
            return ""
        
        # Converte in stringa e minuscolo subito
        text = str(text).lower()
        
        # RIMUOVE emoji completamente (NON converte in testo)
        emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"  # dingbats
                                  u"\U000024C2-\U0001F251"  # enclosed characters
                                  "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' ', text)  # Sostituisce con spazio
        
        # Rimuove URL e link
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'bit\.ly/\S+', '', text)
        
        # Rimuove menzioni (@user) completamente
        text = re.sub(r'@\w+', '', text)
        
        # Rimuove hashtag completamente (# e testo)
        text = re.sub(r'#\w+', '', text)
        
        # Rimuove RT (retweet)
        text = re.sub(r'\brt\b', '', text)
        
        # Rimuove numeri e caratteri speciali
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Rimuove spazi multipli
        text = re.sub(r'\s+', ' ', text)
        
        # Rimuove spazi all'inizio e fine
        text = text.strip()
        
        # Filtra parole troppo corte o troppo lunghe
        tokens = word_tokenize(text)
        tokens = [
            word for word in tokens 
            if (
                word not in self.stop_words and 
                len(word) >= 3 and 
                len(word) <= 15 and
                word.isalpha()  # Solo lettere alfabetiche
            )
        ]
        
        return ' '.join(tokens)
    
    def prepare_dataset(self, max_samples=50000):
        """Prepara il dataset utilizzando dati pubblici puliti"""
        print("Caricamento del dataset...")
        
        # Utilizziamo il dataset "emotion" che contiene tweet etichettati
        try:
            dataset = load_dataset("emotion")
            
            # Prendiamo solo train e test
            train_data = dataset['train']
            test_data = dataset['test']
            
            # Convertiamo in DataFrame
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            # Combiniamo train e test
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Mappiamo le emozioni a sentiment positivo/negativo
            # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
            positive_emotions = [1, 2, 5]  # joy, love, surprise
            df['sentiment'] = df['label'].apply(lambda x: 1 if x in positive_emotions else 0)
            
        except Exception as e:
            print(f"Errore nel caricamento del dataset emotion: {e}")
            print("Creo un dataset sintetico per testing...")
            
            # Dataset sintetico per testing
            positive_tweets = [
                "I love this beautiful day! So happy and grateful",
                "Amazing news! Just got the job I wanted",
                "Beautiful sunset tonight, feeling blessed",
                "Great movie, highly recommend it to everyone",
                "Just had the best coffee ever! Perfect start",
                "Wonderful time with family and friends today",
                "Accomplished my goals for this week, feeling proud",
                "Such a lovely weather outside, perfect for walking"
            ] * (max_samples // 16)
            
            negative_tweets = [
                "Really disappointed with the service today",
                "Feeling sad and frustrated about everything",
                "Terrible weather ruining my weekend plans",
                "Bad news keeps coming, need a break",
                "Stressed about deadlines and work pressure",
                "Not happy with recent changes at work",
                "Feeling tired and overwhelmed lately",
                "Disappointed in the quality of this product"
            ] * (max_samples // 16)
            
            texts = positive_tweets + negative_tweets
            sentiments = [1] * len(positive_tweets) + [0] * len(negative_tweets)
            
            df = pd.DataFrame({
                'text': texts[:max_samples],
                'sentiment': sentiments[:max_samples]
            })
        
        # Limitiamo il numero di campioni se necessario
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        print(f"Dataset originale: {len(df)} campioni")
        
        # Pulizia del testo
        print("Pulizia del testo...")
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Rimuovi testi vuoti dopo la pulizia
        df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
        
        print(f"Dataset dopo pulizia: {len(df)} campioni")
        print(f"Distribuzione sentiment:")
        print(f"- Positivi: {sum(df['sentiment'] == 1)} ({sum(df['sentiment'] == 1)/len(df)*100:.1f}%)")
        print(f"- Negativi: {sum(df['sentiment'] == 0)} ({sum(df['sentiment'] == 0)/len(df)*100:.1f}%)")
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Divide i dati in train, validation e test"""
        
        # Prima divisione: train+val e test
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['clean_text'], df['sentiment'], 
            test_size=test_size, 
            random_state=42, 
            stratify=df['sentiment']
        )
        
        # Seconda divisione: train e validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=42, 
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Funzione principale per preparare i dati"""
    preprocessor = TweetPreprocessor()
    
    # Prepara il dataset
    df = preprocessor.prepare_dataset(max_samples=20000)
    
    # Divide i dati
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # Salva i dati processati
    os.makedirs('data', exist_ok=True)
    
    pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv('data/train.csv', index=False)
    pd.DataFrame({'text': X_val, 'sentiment': y_val}).to_csv('data/val.csv', index=False)
    pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv('data/test.csv', index=False)
    
    print("\nDati salvati in:")
    print(f"- Training: {len(X_train)} campioni")
    print(f"- Validation: {len(X_val)} campioni") 
    print(f"- Test: {len(X_test)} campioni")
    
    return df

if __name__ == "__main__":
    main() 