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
import logging

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
    Include anche capacità di data augmentation avanzate.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Rimuoviamo alcune negazioni dalle stopwords, perché sono importanti per il sentiment
        self.stop_words.difference_update(["no", "not", "nor", "ain", "aren't", "couldn't", 
                                           "didn't", "doesn't", "hadn't", "hasn't", "haven't", 
                                           "isn't", "mightn't", "mustn't", "needn't", "shan't", 
                                           "shouldn't", "wasn't", "weren't", "won't", "wouldn't"])
        
        # Dizionari per normalizzazione
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "it's": "it is", "that's": "that is",
            "what's": "what is", "where's": "where is", "how's": "how is",
            "there's": "there is", "here's": "here is"
        }
        
        # Intensificatori e modificatori
        self.intensifiers = {
            "very": "<INTENSIFY>", "really": "<INTENSIFY>", "extremely": "<INTENSIFY>",
            "absolutely": "<INTENSIFY>", "totally": "<INTENSIFY>", "completely": "<INTENSIFY>",
            "quite": "<MODERATE>", "somewhat": "<MODERATE>", "fairly": "<MODERATE>",
            "slightly": "<REDUCE>", "barely": "<REDUCE>", "hardly": "<REDUCE>"
        }
        
        # Negazioni
        self.negations = {
            "not", "no", "never", "none", "nothing", "nobody", "nowhere",
            "neither", "nor", "without", "against", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "can't", "cannot", "couldn't",
            "shouldn't", "mustn't", "needn't", "shan't", "hasn't", "haven't",
            "hadn't", "isn't", "aren't", "wasn't", "weren't"
        }

    def expand_contractions(self, text):
        """Espande le contrazioni per una migliore comprensione"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def handle_negations(self, text):
        """Gestisce le negazioni marcandole esplicitamente"""
        tokens = text.split()
        result = []
        negate_next = False
        
        for i, token in enumerate(tokens):
            if token.lower() in self.negations:
                result.append("<NEG>")
                negate_next = True
            elif negate_next and token.isalpha():
                result.append(f"<NEG_{token}>")
                negate_next = False
            else:
                result.append(token)
                
        return " ".join(result)
    
    def clean_text(self, text):
        """
        Applica una pulizia avanzata e normalizzazione al testo di un tweet.
        """
        if pd.isna(text):
            return ""
        
        # 1. Conversione in minuscolo e gestione emoji
        text = emoji.demojize(str(text).lower(), delimiters=(" :", ": "))

        # 2. Espansione contrazioni
        text = self.expand_contractions(text)

        # 3. Normalizzazione parole allungate (es. "soooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        # 4. Gestione punteggiatura ripetuta con sentiment preserving
        text = re.sub(r'(\!){3,}', ' <VERY_EXCITED> ', text)
        text = re.sub(r'(\!){2}', ' <EXCITED> ', text)
        text = re.sub(r'(\?){3,}', ' <VERY_CONFUSED> ', text)
        text = re.sub(r'(\?){2}', ' <CONFUSED> ', text)

        # 5. Rimozione URL e link
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'bit\.ly/\S+', '', text)

        # 6. Rimozione menzioni (@user) ma mantieni il contesto
        text = re.sub(r'@\w+', ' <MENTION> ', text)
        
        # 7. Trasformazione hashtag in testo normale
        text = re.sub(r'#(\w+)', r'\1', text)

        # 8. Gestione intensificatori
        for intensifier, token in self.intensifiers.items():
            text = re.sub(rf'\b{intensifier}\b', token, text)

        # 9. Gestione negazioni avanzata
        text = self.handle_negations(text)

        # 10. Rimuove caratteri speciali indesiderati, mantenendo parole, numeri, e i nostri token
        # Questa regex è più permissiva e non rimuove parole valide come 'crap'
        text = re.sub(r'[^\w\s<>_]', ' ', text)

        # 11. Tokenizzazione
        tokens = word_tokenize(text)

        # 12. Filtro avanzato dei token
        cleaned_tokens = []
        for word in tokens:
            if (
                word not in self.stop_words and
                len(word) >= 2 and
                (word.isalpha() or '<' in word) # Manteniamo i token speciali
            ):
                cleaned_tokens.append(word)

        # 13. Rimuove spazi multipli e ricompone
        return " ".join(cleaned_tokens).strip()
    
    def augment_text_random_deletion(self, text, p=0.1):
        """Applica la data augmentation tramite cancellazione casuale di parole."""
        tokens = text.split()
        if len(tokens) <= 1:
            return text
        
        remaining = [token for token in tokens if random.random() > p]
        
        if len(remaining) == 0:
            return tokens[random.randint(0, len(tokens)-1)]
            
        return ' '.join(remaining)

    def augment_text_synonym_replacement(self, text, n=1):
        """Sostituisce n parole casuali con sinonimi semplici"""
        synonyms_dict = {
            'good': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
            'happy': ['joyful', 'cheerful', 'pleased', 'delighted', 'glad'],
            'sad': ['unhappy', 'depressed', 'sorrowful', 'miserable', 'down'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
            'hate': ['despise', 'loathe', 'detest', 'dislike', 'abhor'],
            'nice': ['pleasant', 'lovely', 'beautiful', 'gorgeous', 'attractive'],
            'ugly': ['hideous', 'unattractive', 'repulsive', 'gross', 'disgusting']
        }
        
        tokens = text.split()
        new_tokens = tokens.copy()
        
        replaceable_indices = [i for i, token in enumerate(tokens) 
                              if token.lower() in synonyms_dict]
        
        if not replaceable_indices:
            return text
            
        indices_to_replace = random.sample(replaceable_indices, 
                                         min(n, len(replaceable_indices)))
        
        for idx in indices_to_replace:
            word = tokens[idx].lower()
            if word in synonyms_dict:
                new_tokens[idx] = random.choice(synonyms_dict[word])
                
        return ' '.join(new_tokens)

    def augment_text_random_insertion(self, text, n=1):
        """Inserisce n parole casuali in posizioni casuali"""
        tokens = text.split()
        if len(tokens) < 2:
            return text
            
        insert_words = ['really', 'very', 'quite', 'somewhat', 'totally', 'completely']
        
        for _ in range(n):
            insert_word = random.choice(insert_words)
            insert_position = random.randint(0, len(tokens))
            tokens.insert(insert_position, insert_word)
        
        return ' '.join(tokens)
    
    def augment_text_random_swap(self, text, n=1):
        """Scambia n coppie di parole adiacenti"""
        tokens = text.split()
        if len(tokens) < 2:
            return text
            
        new_tokens = tokens.copy()
        
        for _ in range(n):
            idx = random.randint(0, len(new_tokens) - 2)
            new_tokens[idx], new_tokens[idx + 1] = new_tokens[idx + 1], new_tokens[idx]
            
        return ' '.join(new_tokens)

    def augment_text_advanced(self, text, techniques=['deletion', 'synonym', 'insertion']):
        """Applica tecniche di augmentation multiple in modo casuale"""
        if not techniques:
            return text
            
        technique = random.choice(techniques)
        
        if technique == 'deletion':
            return self.augment_text_random_deletion(text, p=0.1)
        elif technique == 'synonym':
            return self.augment_text_synonym_replacement(text, n=1)
        elif technique == 'insertion':
            return self.augment_text_random_insertion(text, n=1)
        elif technique == 'swap':
            return self.augment_text_random_swap(text, n=1)
        else:
            return text

    def load_multiple_datasets(self):
        """Carica e combina multiple fonti di dataset per maggiore diversità"""
        print("🔄 Caricamento di multiple fonti di dataset...")
        datasets = []
        
        # 1. Dataset Emotion
        try:
            print("   📥 Caricando dataset 'emotion'...")
            emotion_dataset = load_dataset("emotion")
            emotion_df = pd.concat([
                pd.DataFrame(emotion_dataset['train']),
                pd.DataFrame(emotion_dataset['test']),
                pd.DataFrame(emotion_dataset['validation'])
            ]).reset_index(drop=True)
            
            # Mapping delle emozioni: 0-2-4 = negativo, 1-3-5 = positivo
            positive_emotions = [1, 2, 5]  # joy, love, surprise
            emotion_df['sentiment'] = emotion_df['label'].apply(
                lambda x: 1 if x in positive_emotions else 0
            )
            emotion_df['source'] = 'emotion'
            datasets.append(emotion_df[['text', 'sentiment', 'source']])
            print(f"      ✅ Caricati {len(emotion_df)} campioni da 'emotion'")
        except Exception as e:
            print(f"      ❌ Errore nel caricare 'emotion': {e}")
        
        # 2. Dataset Tweet Eval (se disponibile)
        try:
            print("   📥 Caricando dataset 'tweet_eval' sentiment...")
            tweet_eval = load_dataset("tweet_eval", "sentiment")
            tweet_df = pd.concat([
                pd.DataFrame(tweet_eval['train']),
                pd.DataFrame(tweet_eval['test']),
                pd.DataFrame(tweet_eval['validation'])
            ]).reset_index(drop=True)
            
            # tweet_eval ha già sentiment 0,1,2 -> mappiamo a binario
            tweet_df['sentiment'] = tweet_df['label'].apply(lambda x: 0 if x == 0 else 1)
            tweet_df['source'] = 'tweet_eval'
            datasets.append(tweet_df[['text', 'sentiment', 'source']])
            print(f"      ✅ Caricati {len(tweet_df)} campioni da 'tweet_eval'")
        except Exception as e:
            print(f"      ⚠️ 'tweet_eval' non disponibile: {e}")
        
        # 3. Dataset IMDB (per varietà di linguaggio)
        try:
            print("   📥 Caricando campioni da 'imdb'...")
            imdb_dataset = load_dataset("imdb")
            # Prendiamo solo un subset per bilanciare
            imdb_subset = pd.concat([
                pd.DataFrame(imdb_dataset['train']).sample(n=5000, random_state=42),
                pd.DataFrame(imdb_dataset['test']).sample(n=2000, random_state=42)
            ]).reset_index(drop=True)
            
            imdb_subset['source'] = 'imdb'
            # IMDB ha già sentiment 0,1
            datasets.append(imdb_subset[['text', 'label', 'source']].rename(columns={'label': 'sentiment'}))
            print(f"      ✅ Caricati {len(imdb_subset)} campioni da 'imdb'")
        except Exception as e:
            print(f"      ⚠️ 'imdb' non disponibile: {e}")
        
        if not datasets:
            raise Exception("❌ Nessun dataset è stato caricato con successo!")
        
        # Combina tutti i dataset
        combined_df = pd.concat(datasets, ignore_index=True)
        
        print(f"\n📊 Dataset combinato: {len(combined_df)} campioni totali")
        print("   Distribuzione per fonte:")
        print(combined_df['source'].value_counts())
        print("   Distribuzione sentiment:")
        print(combined_df['sentiment'].value_counts(normalize=True))
        
        return combined_df

    def prepare_and_augment_dataset(self, augment_minority=True, aug_factor=2, use_multiple_sources=True):
        """
        Prepara il dataset usando multiple fonti, li pulisce con il metodo avanzato
        e applica augmentation sofisticata.
        """
        if use_multiple_sources:
            print("🔄 Preparazione dataset multi-fonte...")
            df = self.load_multiple_datasets()
        else:
            print("🔄 Preparazione dataset singola fonte...")
            dataset = load_dataset("emotion")
            df = pd.concat([
                pd.DataFrame(dataset['train']),
                pd.DataFrame(dataset['test']),
                pd.DataFrame(dataset['validation'])
            ]).reset_index(drop=True)

            positive_emotions = [1, 2, 5]
            df['sentiment'] = df['label'].apply(lambda x: 1 if x in positive_emotions else 0)
            df['source'] = 'emotion_only'
        
        print("\n🧹 Pulizia avanzata del testo...")
        df['text'] = df['text'].apply(self.clean_text)
        
        # Rimuovi testi vuoti
        initial_count = len(df)
        df = df[df['text'].str.len() > 0].reset_index(drop=True)
        print(f"   Rimossi {initial_count - len(df)} testi vuoti")
        
        print(f"\n📊 Dataset pulito. Distribuzione classi prima dell'augmentation:")
        print(df['sentiment'].value_counts(normalize=True))

        if augment_minority:
            print(f"\n🔄 Avvio data augmentation avanzata (fattore: {aug_factor})...")
            
            sentiment_counts = df['sentiment'].value_counts()
            minority_class = sentiment_counts.idxmin()
            minority_size = sentiment_counts.min()
            majority_size = sentiment_counts.max()
            
            num_to_generate = int(min(majority_size - minority_size, minority_size * aug_factor))
            
            if num_to_generate > 0:
                minority_df = df[df['sentiment'] == minority_class]
                
                augmented_data = []
                samples_to_augment = minority_df.sample(n=num_to_generate, replace=True, random_state=42)
                
                print(f"   Generando {num_to_generate} nuovi campioni...")
                
                for _, row in samples_to_augment.iterrows():
                    # Applica augmentation avanzata con tecniche multiple
                    augmented_text = self.augment_text_advanced(
                        row['text'], 
                        techniques=['deletion', 'synonym', 'insertion', 'swap']
                    )
                    
                    augmented_data.append({
                        'text': augmented_text,
                        'sentiment': minority_class,
                        'source': f"{row.get('source', 'unknown')}_augmented"
                    })
                
                augmented_df = pd.DataFrame(augmented_data)
                df = pd.concat([df, augmented_df], ignore_index=True)
                
                print(f"   ✅ Generati {num_to_generate} nuovi campioni per la classe {minority_class}")
                print(f"\n📊 Distribuzione finale:")
                print(df['sentiment'].value_counts(normalize=True))
                if 'source' in df.columns:
                    print("\n   Distribuzione per fonte:")
                    print(df['source'].value_counts())

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

def intelligent_truncate(tokens, max_length, strategy='simple', head_tail_ratio=0.5):
    """
    Applica una strategia di troncamento intelligente a una lista di token,
    assicurando che l'output non superi mai max_length.
    """
    if len(tokens) <= max_length:
        return tokens

    if strategy == 'head+tail':
        # Calcola quanti token tenere dall'inizio e dalla fine
        tail_count = int(max_length * (1 - head_tail_ratio))
        head_count = max_length - tail_count
        
        # Prendi le due parti e le unisce
        head = tokens[:head_count]
        tail = tokens[-tail_count:]
        return head + tail
    
    # La strategia di default è il troncamento semplice
    return tokens[:max_length] 