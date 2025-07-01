# 🐦 Tweet Sentiment Analysis con Rete Neurale

Questo progetto implementa una rete neurale LSTM per l'analisi del sentiment dei tweet, classificandoli come **positivi** o **negativi**.

## 🚀 Caratteristiche

- **Rete Neurale LSTM Bidirezionale** per catturare dipendenze a lungo termine nel testo
- **Preprocessing automatico** dei tweet (rimozione URL, menzioni, emoji processing)
- **Dataset puliti** utilizzando dataset pubblici di qualità
- **Vocabolario dinamico** costruito automaticamente dai dati
- **Visualizzazione dell'addestramento** con grafici di loss e accuracy
- **Modalità di test interattiva** per testare nuovi tweet
- **Supporto GPU** per addestramento accelerato

## 📋 Requisiti

```bash
pip install -r requirements.txt
```

### Dipendenze principali:
- PyTorch (>= 2.0.0)
- Transformers
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn

## 🏗️ Struttura del Progetto

```
tweet-ML/
├── data_preparation.py      # Preprocessing e preparazione dati
├── sentiment_model.py       # Architettura della rete neurale
├── train_sentiment_model.py # Script di addestramento
├── test_model.py           # Script per testare il modello
├── requirements.txt        # Dipendenze
├── README.md              # Questo file
├── data/                  # Dataset processati (generato automaticamente)
└── models/                # Modelli addestrati (generato automaticamente)
```

## 🎯 Come Utilizzare

### 1. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Addestramento del Modello

```bash
python train_sentiment_model.py
```

Questo script:
- Scarica e prepara un dataset pulito di tweet
- Costruisce il vocabolario automaticamente
- Addestra una rete LSTM bidirezionale
- Salva il miglior modello e i grafici di addestramento

**Parametri di default:**
- 20.000 campioni di addestramento
- 8 epochs
- LSTM con 64 hidden units, 2 layers
- Embedding dimension: 128
- Dropout: 0.3

### 3. Test del Modello

```bash
python test_model.py
```

Il script offre 3 modalità:

1. **Test su esempi predefiniti**: Testa il modello su tweet di esempio
2. **Modalità interattiva**: Inserisci tweet manualmente per testare
3. **Analisi file CSV**: Analizza tweet da un file CSV

## 🧠 Architettura della Rete Neurale

```
Input Tweet → Preprocessing → Tokenization → Embedding Layer
                                                    ↓
                                           LSTM Bidirezionale (2 layers)
                                                    ↓
                                           Fully Connected Layers
                                                    ↓
                                           Output: [Negativo, Positivo]
```

### Dettagli Tecnici:
- **Embedding Layer**: Converte token in vettori densi (dim: 128)
- **LSTM Bidirezionale**: Cattura dipendenze in entrambe le direzioni
- **Dropout**: Previene overfitting (30%)
- **Classificatore**: Tre layer fully connected con ReLU

## 📊 Preprocessing dei Dati

Il preprocessing include:

1. **Conversione emoji** in testo descrittivo
2. **Rimozione URL** e link
3. **Pulizia menzioni** (@user) e hashtag
4. **Rimozione caratteri speciali** e numeri
5. **Tokenizzazione** e rimozione stopwords
6. **Normalizzazione** (lowercase)

## 🎮 Esempi di Utilizzo

### Test Interattivo
```bash
python test_model.py
# Scegli opzione 2 per modalità interattiva
```

### Test su CSV
```python
# Prepara un CSV con colonna 'text' contenente i tweet
# Esegui:
python test_model.py
# Scegli opzione 3 e inserisci il percorso del CSV
```

### Programmatico
```python
from test_model import TweetSentimentPredictor

# Carica il predictor
predictor = TweetSentimentPredictor()

# Predici un singolo tweet
sentiment, confidence, clean_text = predictor.predict_single(
    "I love this beautiful day! So happy! 😊"
)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}")

# Predici multipli tweet
tweets = ["Great movie!", "Terrible service!"]
results = predictor.predict_batch(tweets)
print(results)
```

## 📈 Risultati Attesi

Con il dataset di default, il modello dovrebbe raggiungere:
- **Accuracy di training**: ~85-90%
- **Accuracy di validazione**: ~80-85%
- **Accuracy di test**: ~80-85%

I risultati vengono salvati in:
- `models/best_model.pth`: Miglior modello addestrato
- `models/vocabulary.pkl`: Vocabolario per preprocessing
- `models/training_history.png`: Grafici di loss e accuracy

## 🔧 Personalizzazione

### Modificare i Parametri del Modello

Edita i parametri in `train_sentiment_model.py`:

```python
# Parametri modificabili
MAX_SAMPLES = 50000      # Più dati = miglior performance
NUM_EPOCHS = 15          # Più epochs = miglior addestramento
EMBEDDING_DIM = 256      # Dimensione embedding
HIDDEN_DIM = 128         # Dimensione hidden LSTM
LEARNING_RATE = 0.0005   # Learning rate
```

### Utilizzare Dataset Personalizzati

Modifica la funzione `prepare_dataset()` in `data_preparation.py` per caricare i tuoi dati.

## 🛠️ Troubleshooting

### Errore "Dataset non trovato"
Il modello utilizza dataset pubblici. Se il download fallisce, verrà usato un dataset sintetico per testing.

### Memoria insufficiente
Riduci:
- `BATCH_SIZE` (da 32 a 16)
- `MAX_SAMPLES` (da 20000 a 10000)
- `HIDDEN_DIM` (da 64 a 32)

### GPU non rilevata
Il modello funziona anche su CPU. Per usare GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Dataset Utilizzati

Il progetto utilizza dataset pubblici come:
- **Emotion Dataset** (HuggingFace): Tweet etichettati con emozioni
- **Dataset sintetico** come fallback per testing

## 🤝 Contributi

1. Fork il repository
2. Crea un branch per la tua feature
3. Commit le modifiche
4. Push e apri una Pull Request

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT.

---

**Buon addestramento! 🚀** 