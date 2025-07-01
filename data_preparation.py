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
import random

# Download delle risorse NLTK necessarie, in modo esplicito e robusto
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}.zip')
    except LookupError:
         try:
            nltk.data.find(f'corpora/{resource}.zip')
         except LookupError:
            print(f"Risorsa NLTK '{resource}' non trovata. Download in corso...")
            nltk.download(resource)

class AdvancedTweetPreprocessor:
    """
    Esegue un preprocessing avanzato dei tweet, interpretando e normalizzando
    il linguaggio dei social media invece di rimuoverlo solamente.
    Include anche capacità di data augmentation.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Rimuoviamo alcune negazioni dalle stopwords, perché sono importanti per il sentiment
        self.stop_words.difference_update(["no", "not", "nor", "ain", "aren't", "couldn't", 
                                           "didn't", "doesn't", "hadn't", "hasn't", "haven't", 
                                           "isn't", "mightn't", "mustn't", "needn't", "shan't", 
                                           "shouldn't", "wasn't", "weren't", "won't", "wouldn't"])

    def clean_text(self, text):
        """
        Applica una pulizia avanzata e normalizzazione al testo di un tweet.
        """
        if pd.isna(text):
            return ""

        # 1. Conversione in minuscolo e gestione emoji
        text = emoji.demojize(str(text).lower(), delimiters=(" :", ": "))

        # 2. Normalizzazione parole allungate (es. "soooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        # 3. Gestione punteggiatura ripetuta
        text = re.sub(r'(\!){2,}', ' <repeated_exc> ', text)
        text = re.sub(r'(\?){2,}', ' <repeated_qst> ', text)

        # 4. Rimozione URL e link
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'bit\.ly/\S+', '', text)

        # 5. Rimozione menzioni (@user)
        text = re.sub(r'@\w+', '', text)
        
        # 6. Trasformazione hashtag in testo normale
        text = re.sub(r'#(\w+)', r'\1', text)

        # 7. Rimuove caratteri non-alfanumerici (mantenendo gli spazi e i token speciali)
        text = re.sub(r'[^a-zA-Z\s_<>]+', ' ', text)

        # 8. Tokenizzazione
        tokens = word_tokenize(text)

        # 9. Filtro avanzato dei token
        cleaned_tokens = []
        for word in tokens:
            if (
                word not in self.stop_words and
                len(word) >= 2 and
                (word.isalpha() or '<' in word) # Manteniamo i token speciali
            ):
                cleaned_tokens.append(word)

        # 10. Rimuove spazi multipli e ricompone
        return " ".join(cleaned_tokens).strip()
    
    def augment_text_random_deletion(self, text, p=0.1):
        """
        Applica la data augmentation tramite cancellazione casuale di parole.
        'p' è la probabilità che una parola venga cancellata.
        """
        tokens = text.split()
        if len(tokens) <= 1:
            return text
        
        remaining = [token for token in tokens if random.random() > p]
        
        if len(remaining) == 0:
            return tokens[random.randint(0, len(tokens)-1)] # Ritorna una parola a caso se tutto viene cancellato
            
        return ' '.join(remaining)

    def prepare_and_augment_dataset(self, augment_minority=True, aug_factor=1):
        """
        Prepara il dataset usando dati pubblici, li pulisce con il metodo avanzato
        e opzionalmente aumenta la classe minoritaria.
        """
        print("Caricamento del dataset 'emotion'...")
        dataset = load_dataset("emotion")
        
        df = pd.concat([
            pd.DataFrame(dataset['train']),
            pd.DataFrame(dataset['test']),
            pd.DataFrame(dataset['validation'])
        ]).reset_index(drop=True)

        # 0: sadness(0), anger(3), fear(4) -> NEGATIVO (0)
        # 1: joy(1), love(2), surprise(5) -> POSITIVO (1)
        positive_emotions = [1, 2, 5]
        df['sentiment'] = df['label'].apply(lambda x: 1 if x in positive_emotions else 0)
        
        print("Pulizia del testo con AdvancedTweetPreprocessor...")
        df['text'] = df['text'].apply(self.clean_text)
        
        df.drop(df[df.text == ''].index, inplace=True)
        
        print(f"Dataset pulito. Distribuzione classi prima dell'augmentation:")
        print(df['sentiment'].value_counts(normalize=True))

        if augment_minority:
            print("Avvio della data augmentation (Random Deletion) per la classe minoritaria...")
            
            sentiment_counts = df['sentiment'].value_counts()
            minority_class = sentiment_counts.idxmin()
            minority_size = sentiment_counts.min()
            majority_size = sentiment_counts.max()
            
            num_to_generate = int(min(majority_size - minority_size, minority_size * aug_factor))
            
            if num_to_generate > 0:
                minority_df = df[df['sentiment'] == minority_class]
                
                augmented_texts = []
                samples_to_augment = minority_df.sample(n=num_to_generate, replace=True, random_state=42)
                
                for text in samples_to_augment['text']:
                    augmented_texts.append({
                        'text': self.augment_text_random_deletion(text, p=0.15),
                        'sentiment': minority_class,
                        'label': -1 
                    })
                
                augmented_df = pd.DataFrame(augmented_texts)
                df = pd.concat([df, augmented_df], ignore_index=True)
                
                print(f"Generati {num_to_generate} nuovi campioni per la classe {minority_class}.")
                print("Distribuzione classi dopo l'augmentation:")
                print(df['sentiment'].value_counts(normalize=True))

        return df.sample(frac=1, random_state=42).reset_index(drop=True)


# Funzione di divisione dati, ora indipendente
def split_data(df, test_size=0.2, val_size=0.1):
    """Divide un DataFrame in set di training, validazione e test."""
    
    # Assicurati che le colonne 'text' e 'sentiment' esistano
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Il DataFrame deve contenere le colonne 'text' e 'sentiment'")

    X = df['text'].values
    y = df['sentiment'].values

    # Primo split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Calcola la dimensione del set di validazione relativa al set train_val
    relative_val_size = val_size / (1 - test_size)

    # Secondo split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val
    )
    
    # Salvataggio
    os.makedirs('data', exist_ok=True)
    pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv('data/train.csv', index=False)
    pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv('data/test.csv', index=False)
    pd.DataFrame({'text': X_val, 'sentiment': y_val}).to_csv('data/val.csv', index=False)
    
    print(f"Dati divisi e salvati: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test.")


def main():
    # Usiamo il nuovo pre-processore avanzato
    preprocessor = AdvancedTweetPreprocessor()
    
    # Prepara e aumenta il dataset
    augmented_df = preprocessor.prepare_and_augment_dataset(augment_minority=True)
    
    print("\nEsempi di testo dopo la pulizia avanzata:")
    print(augmented_df[['text', 'sentiment']].head())
    
    print("\nEsempi di testo aumentato (se generato):")
    if 'label' in augmented_df.columns:
        augmented_samples = augmented_df[augmented_df['label'] == -1]
        if not augmented_samples.empty:
            print(augmented_samples[['text', 'sentiment']].head())
        else:
            print("Nessun testo aumentato da mostrare.")

    print("\nDivisione dei dati in train, validation, test...")
    split_data(augmented_df)
    
    print("\nFile CSV generati con successo in 'data/':")
    print("- train.csv")
    print("- val.csv")
    print("- test.csv")


if __name__ == '__main__':
    main() 