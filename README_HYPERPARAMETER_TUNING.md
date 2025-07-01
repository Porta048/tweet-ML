# 🔧 Sistema di Hyperparameter Tuning - Tweet Sentiment Analysis

## 📋 Panoramica

Il sistema è stato completamente ridisegnato per supportare **configurazioni esterne** e **hyperparameter tuning automatico**. Non più parametri hardcoded nel codice!

### ✨ Funzionalità Principali

- 🎛️ **Configurazioni YAML**: Tutti i parametri in file esterni modificabili
- 🔬 **Hyperparameter Tuning**: Optuna, Grid Search, Random Search
- 🚀 **Auto-ottimizzazione**: Adattamento automatico alla memoria disponibile
- 📊 **Analisi risultati**: Report dettagliati e visualizzazioni
- 🎯 **Riproducibilità**: Seed e determinismo controllato
- 💾 **Gestione esperimenti**: Tracking e confronto automatico

## 🏗️ Struttura File

```
tweet-ML/
├── config.yaml                    # Configurazione principale
├── config_dev.yaml               # Configurazione sviluppo (test rapidi)
├── config_manager.py             # Gestore configurazioni
├── hyperparameter_tuning.py      # Sistema tuning
├── run_hyperparameter_tuning.py  # Launcher principale
├── requirements.txt              # Dipendenze aggiornate
└── results/                      # Directory risultati esperimenti
    └── hyperparameter_tuning/    # Risultati tuning
```

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Validazione Configurazione

```bash
python run_hyperparameter_tuning.py --config config.yaml --validate
```

### 3. Training Singolo

```bash
# Training con configurazione di default
python run_hyperparameter_tuning.py --config config.yaml --mode training

# Training veloce per test
python run_hyperparameter_tuning.py --config config_dev.yaml --mode training
```

### 4. Hyperparameter Tuning

```bash
# Random search veloce (15 trial)
python run_hyperparameter_tuning.py --config config_dev.yaml --mode tuning

# Optuna completo (100 trial)
python run_hyperparameter_tuning.py --config config.yaml --mode tuning --method optuna --trials 100

# Grid search
python run_hyperparameter_tuning.py --config config.yaml --mode tuning --method grid_search
```

## 📊 Metodi di Hyperparameter Tuning

### 🎲 Random Search
- **Veloce** per esplorazioni iniziali
- **Configurabile** numero di campioni casuali
- **Ideale** per spazi di ricerca grandi

```yaml
hyperparameter_tuning:
  method: "random_search"
  n_random_samples: 50
```

### 🔬 Optuna (Raccomandato)
- **Bayesian optimization** intelligente
- **Pruning automatico** di trial poco promettenti
- **Visualizzazioni avanzate** dei risultati

```yaml
hyperparameter_tuning:
  method: "optuna"
  n_trials: 100
  timeout: 7200  # 2 ore
```

### 🏗️ Grid Search
- **Esplorazione sistematica** di tutte le combinazioni
- **Completezza garantita** (ma più lento)
- **Ideale** per spazi di ricerca piccoli

```yaml
hyperparameter_tuning:
  method: "grid_search"
  grid_search:
    embedding_dim: [64, 128]
    hidden_dim: [32, 64]
    learning_rate: [0.001, 0.01]
```

## ⚙️ Configurazione YAML

### Struttura Principale

```yaml
# Informazioni progetto
project:
  name: "tweet-sentiment-analysis"
  version: "2.0.0"

# Dataset
data:
  max_samples: 20000
  max_length: 100
  train_split: 0.7

# Architettura modello
model:
  embedding_dim: 128
  hidden_dim: 64
  num_layers: 2
  use_attention: true

# Training
training:
  num_epochs: 500
  batch_size: 32
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 50

# Hyperparameter tuning
hyperparameter_tuning:
  enabled: true
  method: "optuna"
  n_trials: 100
```

### 🎛️ Parametri Configurabili

#### **Architettura Modello**
- `embedding_dim`: Dimensione embeddings (32, 64, 128, 256)
- `hidden_dim`: Dimensione LSTM hidden states
- `num_layers`: Numero layer LSTM (1-3)
- `dropout`: Dropout rate (0.1-0.5)
- `use_attention`: Attention mechanism (true/false)

#### **Training**
- `learning_rate`: Learning rate (0.0001-0.01)
- `batch_size`: Dimensione batch (8, 16, 32, 64)
- `optimizer_type`: Ottimizzatore (adam, adamw, sgd)
- `weight_decay`: L2 regularization (1e-6 to 1e-4)

#### **Hardware**
- `auto_memory_management`: Ottimizzazione automatica
- `dynamic_batch_size`: Batch size adattivo
- `mixed_precision`: Mixed precision training

## 📈 Risultati e Analisi

### Output del Tuning

```
🏆 RISULTATI HYPERPARAMETER TUNING
┌─────────────────┬─────────────┐
│ Metrica         │ Valore      │
├─────────────────┼─────────────┤
│ Metodo          │ OPTUNA      │
│ Miglior Score   │ 0.8642      │
│ Durata          │ 1.2h        │
│ Total Trials    │ 100         │
│ Successful      │ 95          │
└─────────────────┴─────────────┘

🔧 MIGLIORI PARAMETRI
┌─────────────────┬─────────────┐
│ Parametro       │ Valore      │
├─────────────────┼─────────────┤
│ embedding_dim   │ 128         │
│ hidden_dim      │ 64          │
│ learning_rate   │ 0.001       │
│ dropout         │ 0.3         │
│ use_attention   │ True        │
└─────────────────┴─────────────┘
```

### File Generati

```
results/hyperparameter_tuning/session_20240101_120000/
├── tuning_results.json          # Risultati principali
├── trials_database.json         # Database completo trial
├── trials_analysis.csv          # Dati per analisi
├── analysis_report.md           # Report dettagliato
├── best_config_trial_42.yaml    # Miglior configurazione
├── optimization_history.html    # Visualizzazione Optuna
├── param_importances.html       # Importanza parametri
└── parallel_coordinate.html     # Coordinate parallele
```

## 🔧 Uso Avanzato

### Configurazioni Multiple

```bash
# Crea configurazioni personalizzate
cp config.yaml config_gpu.yaml      # Per GPU potenti
cp config.yaml config_production.yaml  # Per produzione

# Modifica parametri specifici
# config_gpu.yaml: batch_size: 64, mixed_precision: true
# config_production.yaml: num_epochs: 1000, n_trials: 500
```

### Spazio di Ricerca Personalizzato

```yaml
hyperparameter_tuning:
  search_space:
    embedding_dim: [64, 128, 256, 512]     # Più opzioni
    hidden_dim: [32, 64, 128, 256, 512]    # Range esteso
    learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01]  # Più granularità
    dropout: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]      # Fine-grained
    use_attention: [true, false]           # Confronto attention
    classifier_layers: [                   # Architetture diverse
      [64, 32],
      [128, 64, 32], 
      [256, 128, 64],
      [512, 256, 128, 64]
    ]
```

### Early Stopping Intelligente

```yaml
training:
  early_stopping:
    enabled: true
    patience: 50                 # Epoche senza miglioramento
    metric: "f1_score"          # Metrica da monitorare
    min_delta: 0.0001           # Miglioramento minimo
```

### Memoria Dinamica

```yaml
hardware:
  auto_memory_management: true   # Auto-ottimizzazione
  memory_threshold: 0.8         # Usa max 80% memoria
  dynamic_batch_size: true      # Adatta batch size
  min_batch_size: 4             # Limite minimo
  max_batch_size: 64            # Limite massimo
```

## 🎯 Best Practices

### 1. **Sviluppo Iterativo**

```bash
# 1. Test rapido con config_dev.yaml
python run_hyperparameter_tuning.py --config config_dev.yaml --mode tuning --trials 10

# 2. Affina parametri promettenti
# Modifica config.yaml con i migliori parametri trovati

# 3. Tuning completo
python run_hyperparameter_tuning.py --config config.yaml --mode tuning --method optuna --trials 100
```

### 2. **Gestione Risorse**

```yaml
# Per sistemi con poca memoria
hardware:
  memory_threshold: 0.6         # Usa solo 60%
  dynamic_batch_size: true
  cleanup_every_n_batches: 20   # Pulizia frequente

# Per GPU potenti
hardware:
  mixed_precision: true         # Dimezza uso memoria
  max_batch_size: 128           # Batch più grandi
```

### 3. **Monitoraggio Progressi**

```bash
# Usa Rich per output colorato e tabelle
pip install rich

# Monitoring in tempo reale
watch -n 30 "ls -la results/hyperparameter_tuning/session_*/tuning_results.json"
```

### 4. **Riproducibilità**

```yaml
reproducibility:
  seed: 42                      # Seed fisso
  deterministic: true           # Operazioni deterministiche
  benchmark: false              # Disabilita ottimizzazioni non-deterministiche
```

## 🐛 Troubleshooting

### Errori Comuni

#### **"Out of Memory"**
```yaml
# Riduci parametri
hardware:
  dynamic_batch_size: true
  max_batch_size: 16
model:
  embedding_dim: 64
  hidden_dim: 32
```

#### **"Trial Failed"**
```bash
# Controlla log dettagliati
tail -f logs/training.log

# Usa configurazione debug
python run_hyperparameter_tuning.py --config config_dev.yaml
```

#### **"Optuna Import Error"**
```bash
# Installa dipendenze complete
pip install optuna plotly kaleido
```

### Performance Tips

1. **SSD Storage**: Salva risultati su SSD per I/O veloce
2. **RAM**: Almeno 8GB per tuning completo
3. **CPU**: Usa `dataloader_workers: 4` su CPU multi-core
4. **GPU**: Abilita `mixed_precision: true` su GPU moderne

## 📊 Esempi di Risultati

### Random Search (15 trial, 10 min)
```
Best F1-Score: 0.8234
Best Parameters:
  embedding_dim: 64
  hidden_dim: 32
  learning_rate: 0.01
  dropout: 0.2
```

### Optuna (100 trial, 2 ore)
```
Best F1-Score: 0.8642
Best Parameters:
  embedding_dim: 128
  hidden_dim: 64
  learning_rate: 0.001
  dropout: 0.3
  use_attention: True
  classifier_layers: [128, 64, 32]
```

### Grid Search (16 combinazioni, 45 min)
```
Best F1-Score: 0.8567
Combinations Tested: 16/16
Complete Coverage: ✅
```

## 🔮 Prossimi Sviluppi

- [ ] **AutoML Integration**: Supporto per AutoGluon/AutoKeras
- [ ] **Multi-GPU**: Training distribuito
- [ ] **Cloud Integration**: AWS SageMaker, Google AI Platform
- [ ] **Advanced Pruning**: Early stopping più intelligente
- [ ] **Neural Architecture Search**: Ottimizzazione architettura automatica

---

## 💡 Suggerimenti

1. **Inizia sempre** con `config_dev.yaml` per test rapidi
2. **Monitora le risorse** durante il tuning
3. **Salva configurazioni** promettenti per esperimenti futuri
4. **Usa Optuna** per ricerche serie, Random Search per esplorazioni
5. **Analizza i report** per capire l'importanza dei parametri

**Buon tuning! 🚀** 