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
            
            # Dataset sintetico molto più ampio e variegato
            positive_tweets = [
                # Gioia e felicità
                "I love this beautiful day! So happy and grateful",
                "Amazing news! Just got the job I wanted",
                "Beautiful sunset tonight, feeling blessed",
                "Great movie, highly recommend it to everyone",
                "Just had the best coffee ever! Perfect start",
                "Wonderful time with family and friends today",
                "Accomplished my goals for this week, feeling proud",
                "Such a lovely weather outside, perfect for walking",
                "Finally finished my project! So excited about the results",
                "Had an incredible vacation! Best trip ever",
                "My favorite song is playing on the radio right now",
                "Just received some fantastic news from my doctor",
                "This restaurant has the most delicious food I have ever tasted",
                "Perfect day for a picnic in the park with friends",
                "Just got promoted at work! Dreams do come true",
                "Beautiful flowers are blooming in my garden today",
                "Found the perfect gift for my mom's birthday",
                "Laughing so hard watching this comedy show",
                "Just finished reading an amazing book that inspired me",
                "Sunny weather makes everything feel so much better",
                
                # Amore e relazioni
                "I absolutely adore my new puppy! So cute and playful",
                "Spending quality time with loved ones is the best",
                "My partner surprised me with a romantic dinner tonight",
                "Grateful for all the amazing people in my life",
                "Love conquers all! Feeling optimistic about everything",
                "My children make me smile every single day",
                "Best friends forever! So lucky to have them",
                "Anniversary dinner was absolutely perfect and romantic",
                "Family reunion was full of love and happiness",
                "My heart is full of joy and gratitude today",
                
                # Successi e risultati
                "Won first place in the competition! Cannot believe it",
                "Successfully completed my training program today",
                "Achieved my fitness goals for this month! Feeling strong",
                "Got accepted into my dream university program",
                "My presentation went perfectly! Everyone loved it",
                "Received an excellent performance review at work",
                "Finally learned how to play my favorite song",
                "Graduated with honors! All the hard work paid off",
                "Started my own business and it's going great",
                "Completed my first marathon! What an achievement",
                
                # Esperienze positive
                "This concert is absolutely incredible! Best night ever",
                "Food at this new restaurant is absolutely delicious",
                "Perfect weather for outdoor activities and adventures",
                "Discovered a hidden gem of a bookstore today",
                "Volunteer work today was so rewarding and meaningful",
                "Morning yoga session left me feeling peaceful",
                "Technology is amazing! This new app is fantastic",
                "Art exhibition was inspiring and thought provoking",
                "Hiking in nature always makes me feel refreshed",
                "Concert tickets arrived! So excited for next week",
                
                # Speranza e ottimismo
                "Tomorrow is going to be an amazing day",
                "Everything will work out perfectly in the end",
                "Life is full of wonderful surprises and opportunities",
                "Feeling hopeful about all the possibilities ahead",
                "Good things are coming my way very soon",
                "Every challenge is an opportunity to grow stronger",
                "Positive thoughts create positive outcomes in life",
                "Universe conspires to help those who believe",
                "Today marks the beginning of something beautiful",
                "Grateful for second chances and new beginnings"
            ]
            
            negative_tweets = [
                # Tristezza e delusione
                "Really disappointed with the service today",
                "Feeling sad and frustrated about everything",
                "Terrible weather ruining my weekend plans",
                "Bad news keeps coming, need a break",
                "Stressed about deadlines and work pressure",
                "Not happy with recent changes at work",
                "Feeling tired and overwhelmed lately",
                "Disappointed in the quality of this product",
                "Lost my keys again! This day keeps getting worse",
                "Failed my driving test for the third time today",
                "Received some really disappointing news about my application",
                "Flight got cancelled and now stuck at airport",
                "Broke my phone screen and it's going to cost fortune",
                "Restaurant got my order completely wrong again",
                "Waited two hours for appointment just to be rescheduled",
                "My favorite show got cancelled after one season",
                "Internet has been down all day during important meeting",
                "Missed the last train home and now stranded",
                "Grocery store was out of everything needed for dinner",
                "Computer crashed and lost all my important work",
                
                # Rabbia e frustrazione
                "Traffic is absolutely horrible today! Going to be late",
                "Customer service representative was incredibly rude",
                "Neighbor's loud music kept me awake all night",
                "Can't believe how expensive everything has become lately",
                "Dealing with bureaucracy is driving me absolutely crazy",
                "People who don't return shopping carts are inconsiderate",
                "Waiting in line for hours just to be told office closed",
                "Construction noise outside my window every morning",
                "Phone battery died right when needed it most",
                "Package delivery was supposed to arrive but never came",
                "Credit card got declined even though have plenty money",
                "Elevator is broken again and live on tenth floor",
                "Printer stopped working right before important deadline",
                "Weather forecast was completely wrong for my outdoor event",
                "Favorite restaurant increased prices by fifty percent",
                
                # Paura e ansia
                "Worried about the results of my medical test",
                "Anxiety about job interview tomorrow is overwhelming",
                "Concerned about my family's financial situation lately",
                "Scared about upcoming surgery next week",
                "Nervous about moving to new city next month",
                "Afraid relationship might not work out long term",
                "Worried about climate change and future generations",
                "Anxious about public speaking presentation tomorrow",
                "Concerned about aging parents living alone",
                "Fearful about potential layoffs at my company",
                
                # Solitudine e isolamento
                "Feeling lonely even when surrounded by people",
                "Nobody seems to understand what going through right now",
                "Missing friends who moved away to different cities",
                "Holidays are especially difficult when you're alone",
                "Social media makes me feel more isolated somehow",
                "Wish had someone to talk to about problems",
                "Feeling disconnected from everyone around me lately",
                "Old friends have all moved on with their lives",
                "Hard to make new connections in this digital age",
                "Sometimes feel like nobody really cares about me",
                
                # Problemi di salute
                "Been dealing with chronic pain for months now",
                "Headaches are getting worse and more frequent lately",
                "Doctor couldn't figure out what's wrong with me",
                "Side effects from medication are really unpleasant",
                "Haven't been sleeping well for weeks now",
                "Allergies are making life miserable this season",
                "Recovery from surgery is taking much longer expected",
                "Mental health has been struggling recently",
                "Fatigue is making it hard to do anything",
                "Worried about family history of serious illness"
            ]
            
            # Creiamo un dataset bilanciato con variazioni
            import random
            random.seed(42)  # Per riproducibilità
            
            # Calcoliamo quanti sample servono per ogni sentiment
            samples_per_sentiment = max_samples // 2
            
            # Se non abbiamo abbastanza frasi, le duplichiamo con variazioni
            all_positive = []
            all_negative = []
            
            # Aggiungiamo tutte le frasi base
            all_positive.extend(positive_tweets)
            all_negative.extend(negative_tweets)
            
            # Se servono più frasi, creiamo variazioni
            variations = [
                lambda x: x + " Really happy about this!",
                lambda x: x + " Absolutely wonderful experience!",
                lambda x: x + " Could not be happier right now!",
                lambda x: "Honestly, " + x.lower(),
                lambda x: "Today, " + x.lower(),
                lambda x: "Just want to say: " + x.lower(),
                lambda x: x + " Best feeling ever!",
                lambda x: x + " Totally awesome!",
                lambda x: x + " So grateful for this!",
                lambda x: "Update: " + x.lower()
            ]
            
            negative_variations = [
                lambda x: x + " This is really frustrating!",
                lambda x: x + " Having a terrible day!",
                lambda x: x + " So disappointed right now!",
                lambda x: "Unfortunately, " + x.lower(),
                lambda x: "Sadly, " + x.lower(),
                lambda x: "Just my luck: " + x.lower(),
                lambda x: x + " Worst day ever!",
                lambda x: x + " So annoying!",
                lambda x: x + " Can't deal with this!",
                lambda x: "Update: " + x.lower()
            ]
            
            # Creiamo variazioni positive se necessario
            while len(all_positive) < samples_per_sentiment:
                for tweet in positive_tweets:
                    if len(all_positive) >= samples_per_sentiment:
                        break
                    variation = random.choice(variations)
                    all_positive.append(variation(tweet))
            
            # Creiamo variazioni negative se necessario
            while len(all_negative) < samples_per_sentiment:
                for tweet in negative_tweets:
                    if len(all_negative) >= samples_per_sentiment:
                        break
                    variation = random.choice(negative_variations)
                    all_negative.append(variation(tweet))
            
            # Tagliamo alle dimensioni esatte e mischiamo
            all_positive = all_positive[:samples_per_sentiment]
            all_negative = all_negative[:samples_per_sentiment]
            
            random.shuffle(all_positive)
            random.shuffle(all_negative)
            
            texts = all_positive + all_negative
            sentiments = [1] * len(all_positive) + [0] * len(all_negative)
            
            # Mischiamo tutto insieme
            combined = list(zip(texts, sentiments))
            random.shuffle(combined)
            texts, sentiments = zip(*combined)
            
            df = pd.DataFrame({
                'text': texts,
                'sentiment': sentiments
            })
            
            print(f"Dataset sintetico creato con {len(all_positive)} frasi positive e {len(all_negative)} negative")
        
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