import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_model import SentimentLSTM, SentimentTrainer, TweetDataset, save_vocabulary
from data_preparation import AdvancedTweetPreprocessor
import os
import torch.nn as nn
import gc
import psutil
import time
import logging

# Import del sistema di configurazioni
try:
    from config_manager import ConfigManager
except ImportError:
    ConfigManager = None
    print("⚠️ ConfigManager non disponibile, uso configurazione legacy")

# Setup logging
logger = logging.getLogger(__name__)

def plot_training_history(history):
    """Visualizza la storia dell'addestramento"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Loss comparison
    ax3.plot(epochs, history['train_losses'], 'b-', linewidth=2)
    ax3.set_title('Training Loss over Time')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    
    # Accuracy comparison
    ax4.plot(epochs, history['val_accuracies'], 'r-', linewidth=2)
    ax4.set_title('Validation Accuracy over Time')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(trainer, test_loader, vocab_to_idx, best_model_path='models/best_model.pth'):
    """Valuta il modello sul test set con metriche complete"""
    print("\n" + "="*60)
    print("🎯 VALUTAZIONE FINALE SUL TEST SET")
    print("="*60)
    
    # Carica il miglior modello
    checkpoint = torch.load(best_model_path, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📥 Modello caricato:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"   Val F1-Score: {checkpoint.get('val_f1', 'N/A'):.4f}")
    if checkpoint.get('val_auc_roc') is not None:
        print(f"   Val AUC-ROC: {checkpoint.get('val_auc_roc'):.4f}")
    
    # Valuta sul test set con metriche complete
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_detailed_metrics, test_metrics_obj = trainer.validate(test_loader, criterion, compute_detailed_metrics=True)
    
    print(f"\n📊 RISULTATI TEST SET:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_detailed_metrics['accuracy']:.4f} ({test_detailed_metrics['accuracy']*100:.2f}%)")
    print(f"   Test F1-Score: {test_detailed_metrics['f1_macro']:.4f}")
    print(f"   Test Precision: {test_detailed_metrics['precision_macro']:.4f}")
    print(f"   Test Recall: {test_detailed_metrics['recall_macro']:.4f}")
    if test_detailed_metrics['auc_roc'] is not None:
        print(f"   Test AUC-ROC: {test_detailed_metrics['auc_roc']:.4f}")
    
    # Report dettagliato
    test_metrics_obj.print_detailed_report("TEST SET FINALE")
    
    # Salva metriche complete del test
    test_metrics_obj.save_metrics_report('models/final_test_metrics.txt', "Test Set Finale")
    
    # Salva visualizzazioni
    os.makedirs('models/plots', exist_ok=True)
    
    try:
        test_metrics_obj.plot_confusion_matrix(save_path='models/plots/confusion_matrix_test.png')
        print("✅ Confusion matrix salvata in: models/plots/confusion_matrix_test.png")
    except Exception as e:
        print(f"⚠️ Errore nel salvare confusion matrix: {e}")
    
    try:
        test_metrics_obj.plot_roc_curve(save_path='models/plots/roc_curve_test.png')
        print("✅ ROC curve salvata in: models/plots/roc_curve_test.png")
    except Exception as e:
        print(f"⚠️ Errore nel salvare ROC curve: {e}")
    
    try:
        test_metrics_obj.plot_precision_recall_curve(save_path='models/plots/pr_curve_test.png')
        print("✅ Precision-Recall curve salvata in: models/plots/pr_curve_test.png")
    except Exception as e:
        print(f"⚠️ Errore nel salvare PR curve: {e}")
    
    # Test su esempi specifici
    print("\n" + "="*50)
    print("🧪 TEST SU ESEMPI SPECIFICI")
    print("="*50)
    
    test_texts = [
        "I love this beautiful sunny day",
        "This is the worst movie I have ever seen",
        "Amazing experience highly recommend",
        "Terrible service very disappointed",
        "Perfect weather for walking outside",
        "Feeling sad and frustrated today",
        "Absolutely fantastic! Best day ever!",
        "Horrible experience, never again",
        "Not bad, could be better though",
        "Moderately satisfied with the results"
    ]
    
    print(f"Analizzando {len(test_texts)} esempi...")
    
    for i, text in enumerate(test_texts, 1):
        # Preprocessing del testo
        preprocessor = AdvancedTweetPreprocessor()
        clean_text = preprocessor.clean_text(text)
        
        sentiment, confidence = trainer.predict(clean_text, vocab_to_idx)
        
        # Emoji per rappresentare il sentiment
        emoji = "😊" if sentiment == "Positivo" else "😔"
        confidence_level = "🔥" if confidence > 0.8 else "👍" if confidence > 0.6 else "🤔"
        
        print(f"{i:2d}. {emoji} '{text}'")
        print(f"     Pulito: '{clean_text}'")
        print(f"     Predizione: {sentiment} {confidence_level} (Conf: {confidence:.4f})")
        print()
    
    print("="*60)
    print("🎉 VALUTAZIONE COMPLETATA!")
    print("📁 File salvati:")
    print("   - models/final_test_metrics.txt: Report dettagliato")
    print("   - models/plots/: Grafici e visualizzazioni")
    print("="*60)
    
    return test_detailed_metrics

def calculate_class_weights(train_file='data/train.csv'):
    """Calcola i pesi delle classi per gestire lo sbilanciamento."""
    print("⚖️ Calcolo dei pesi delle classi...")
    try:
        train_df = pd.read_csv(train_file)
        # Conta il numero di occorrenze per ogni classe
        class_counts = train_df['sentiment'].value_counts()
        
        # Calcola i pesi: peso = totale_campioni / (num_classi * campioni_per_classe)
        num_samples = len(train_df)
        num_classes = len(class_counts)
        
        weights = {i: num_samples / (num_classes * count) for i, count in class_counts.items()}
        
        # Ordina i pesi per indice di classe (0, 1, ...) e converti in tensore
        class_weights_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)
        
        print(f"   Distribuzione classi: {class_counts.to_dict()}")
        print(f"   Pesi calcolati: {class_weights_tensor.tolist()}")
        return class_weights_tensor
    except FileNotFoundError:
        print(f"⚠️ File '{train_file}' non trovato. Salto il calcolo dei pesi.")
        return None
    except Exception as e:
        print(f"❌ Errore nel calcolo dei pesi: {e}. Salto il calcolo.")
        return None

def get_memory_info():
    """Ottiene informazioni sulla memoria disponibile"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_percent': memory.percent
    }

def calculate_optimal_batch_size(vocab_size, seq_length=100, embedding_dim=128, hidden_dim=64, available_memory_gb=4):
    """Calcola il batch size ottimale basato sulla memoria disponibile"""
    
    # Stima approssimativa del consumo di memoria per sample (in MB)
    # Embedding: seq_length * embedding_dim * 4 bytes (float32)
    embedding_memory = seq_length * embedding_dim * 4 / (1024*1024)
    
    # LSTM bidirezionale: seq_length * hidden_dim * 2 * 4 * num_layers
    lstm_memory = seq_length * hidden_dim * 2 * 4 * 2 / (1024*1024)
    
    # Attention mechanism: seq_length * seq_length * 4 (attention weights)
    attention_memory = seq_length * seq_length * 4 / (1024*1024)
    
    # Pooling operations: multiple copies of hidden states
    pooling_memory = hidden_dim * 2 * 4 * 4 / (1024*1024)  # 4 different pooling outputs
    
    # Forward pass per sample (MB)
    memory_per_sample = embedding_memory + lstm_memory + attention_memory + pooling_memory
    
    # Aggiungi overhead per gradients (circa 2x) e batch processing
    memory_per_sample *= 3
    
    # Calcola batch size massimo (usa solo 70% della memoria disponibile)
    usable_memory_mb = available_memory_gb * 1024 * 0.7
    max_batch_size = int(usable_memory_mb / memory_per_sample)
    
    # Limiti pratici
    max_batch_size = max(1, min(max_batch_size, 64))  # Min 1, Max 64
    
    return max_batch_size, memory_per_sample

def optimize_model_for_memory(available_memory_gb):
    """Ottimizza i parametri del modello basandosi sulla memoria disponibile"""
    
    if available_memory_gb >= 8:
        # Sistema con molta memoria
        return {
            'batch_size': 32,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'max_samples': 20000,
            'use_attention': True,
            'gradient_accumulation_steps': 1
        }
    elif available_memory_gb >= 4:
        # Sistema con memoria media
        return {
            'batch_size': 16,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'max_samples': 15000,
            'use_attention': True,
            'gradient_accumulation_steps': 2
        }
    elif available_memory_gb >= 2:
        # Sistema con poca memoria
        return {
            'batch_size': 8,
            'embedding_dim': 64,
            'hidden_dim': 32,
            'max_samples': 10000,
            'use_attention': False,
            'gradient_accumulation_steps': 4
        }
    else:
        # Sistema con memoria molto limitata
        return {
            'batch_size': 4,
            'embedding_dim': 64,
            'hidden_dim': 32,
            'max_samples': 5000,
            'use_attention': False,
            'gradient_accumulation_steps': 8
        }

def train_model_with_config(config: 'ConfigManager') -> dict:
    """
    Addestra il modello usando una configurazione ConfigManager
    Restituisce un dizionario con le metriche finali
    """
    logger.info("🚀 Avvio training con configurazioni YAML")
    start_time = time.time()
    
    try:
        # Header
        print("="*60)
        print("ADDESTRAMENTO CON CONFIGURAZIONI YAML")
        print("="*60)
        
        # Analisi memoria sistema (se auto-management è abilitato)
        if config.hardware.auto_memory_management:
            memory_info = get_memory_info()
            print(f"💾 ANALISI MEMORIA SISTEMA:")
            print(f"   Memoria totale: {memory_info['total_gb']:.1f} GB")
            print(f"   Memoria disponibile: {memory_info['available_gb']:.1f} GB")
            print(f"   Memoria utilizzata: {memory_info['used_percent']:.1f}%")
        
        # Parametri dalle configurazioni
        MAX_SAMPLES = config.data.max_samples
        BATCH_SIZE = config.training.batch_size
        EMBEDDING_DIM = config.model.embedding_dim
        HIDDEN_DIM = config.model.hidden_dim
        USE_ATTENTION = config.model.use_attention
        GRADIENT_ACCUMULATION_STEPS = config.training.gradient_accumulation_steps
        NUM_EPOCHS = config.training.num_epochs
        LEARNING_RATE = config.training.learning_rate
        MAX_LENGTH = config.data.max_length
        NUM_LAYERS = config.model.num_layers
        DROPOUT = config.model.dropout
        
        print(f"\n🔧 PARAMETRI DA CONFIGURAZIONE:")
        print(f"   Max samples: {MAX_SAMPLES:,}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Embedding dim: {EMBEDDING_DIM}")
        print(f"   Hidden dim: {HIDDEN_DIM}")
        print(f"   Num layers: {NUM_LAYERS}")
        print(f"   Dropout: {DROPOUT}")
        print(f"   Learning rate: {LEARNING_RATE}")
        print(f"   Max length: {MAX_LENGTH}")
        print(f"   Epochs: {NUM_EPOCHS}")
        print(f"   Attention: {USE_ATTENTION}")
        print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # 1. Preparazione dei dati
        print("\n1. PREPARAZIONE DEI DATI")
        print("-" * 30)
        
        # Usa directory configurate
        data_dir = config.paths.data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')
        
        # Controlla se i dati esistono già
        if os.path.exists(train_csv):
            print("Caricamento dati esistenti...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
            
            X_train, y_train = train_df['text'], train_df['sentiment']
            X_val, y_val = val_df['text'], val_df['sentiment']
            X_test, y_test = test_df['text'], test_df['sentiment']
        else:
            print("Preparazione nuovi dati...")
            preprocessor = AdvancedTweetPreprocessor()
            df = preprocessor.prepare_and_augment_dataset(augment_minority=True)
            # Usa la funzione split_data indipendente
            from data_preparation import split_data
            split_data(df)
            
            # Ricarica i dati divisi
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv) 
            test_df = pd.read_csv(test_csv)
            
            X_train, y_train = train_df['text'], train_df['sentiment']
            X_val, y_val = val_df['text'], val_df['sentiment']
            X_test, y_test = test_df['text'], test_df['sentiment']
            
            # Salva i dati nelle directory configurate
            pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv(train_csv, index=False)
            pd.DataFrame({'text': X_val, 'sentiment': y_val}).to_csv(val_csv, index=False)
            pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv(test_csv, index=False)
        
        print(f"Dataset preparato:")
        print(f"- Training: {len(X_train)} campioni")
        print(f"- Validation: {len(X_val)} campioni")
        print(f"- Test: {len(X_test)} campioni")
        
        # 2. Costruzione del vocabolario
        print("\n2. COSTRUZIONE VOCABOLARIO")
        print("-" * 30)
        
        # Combina tutti i testi per costruire il vocabolario
        all_texts = pd.concat([X_train, X_val, X_test])
        
        # Inizializza trainer temporaneo per costruire vocabolario
        temp_model = SentimentLSTM(100, use_attention=False)  # vocab_size temporaneo
        trainer = SentimentTrainer(temp_model, device)
        vocab_to_idx, idx_to_vocab = trainer.build_vocabulary(all_texts)
        
        # Salva il vocabolario nella directory configurata
        models_dir = config.paths.models_dir
        os.makedirs(models_dir, exist_ok=True)
        vocab_file = os.path.join(models_dir, config.paths.vocabulary_file)
        save_vocabulary(vocab_to_idx, idx_to_vocab, vocab_file)
        
        # 3. Creazione del modello definitivo
        print("\n3. CREAZIONE MODELLO")
        print("-" * 30)
        
        vocab_size = len(vocab_to_idx)
        model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            use_attention=USE_ATTENTION
        )
        
        trainer = SentimentTrainer(model, device)
        
        print(f"Modello creato:")
        print(f"- Vocab size: {vocab_size}")
        print(f"- Embedding dim: {EMBEDDING_DIM}")
        print(f"- Hidden dim: {HIDDEN_DIM}")
        print(f"- Num layers: {NUM_LAYERS}")
        print(f"- Dropout: {DROPOUT}")
        print(f"- Attention mechanism: {USE_ATTENTION}")
        print(f"- Architettura: LSTM bidirezionale + Attention + Pooling")
        
        # Conta i parametri
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"- Parametri totali: {total_params:,}")
        print(f"- Parametri addestrabili: {trainable_params:,}")
        
        # 4. Creazione dataset e dataloader
        print("\n4. CREAZIONE DATALOADER")
        print("-" * 30)
        
        train_dataset = TweetDataset(X_train, y_train, vocab_to_idx, MAX_LENGTH)
        val_dataset = TweetDataset(X_val, y_val, vocab_to_idx, MAX_LENGTH)
        test_dataset = TweetDataset(X_test, y_test, vocab_to_idx, MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"DataLoader creati:")
        print(f"- Train batches: {len(train_loader)}")
        print(f"- Val batches: {len(val_loader)}")
        print(f"- Test batches: {len(test_loader)}")
        
        # 5. Addestramento
        print("\n5. ADDESTRAMENTO")
        print("-" * 30)
        
        best_model_path = os.path.join(models_dir, config.paths.best_model_file)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            model_save_path=best_model_path,
            models_dir=models_dir
        )
        
        # 6. Visualizzazione risultati (se abilitata)
        if config.evaluation.plot_training_history:
            print("\n6. VISUALIZZAZIONE RISULTATI")
            print("-" * 30)
            plot_training_history(history)
        
        # 7. Valutazione finale (se abilitata)
        final_metrics = {}
        if config.evaluation.save_final_test:
            print("\n7. VALUTAZIONE FINALE")
            print("-" * 30)
            final_metrics = evaluate_model(trainer, test_loader, vocab_to_idx, best_model_path)
        
        # Calcola tempo totale
        end_time = time.time()
        training_time = end_time - start_time
        
        # Prepara risultati per hyperparameter tuning
        # Carica le metriche del miglior modello
        best_model_path = os.path.join(models_dir, config.paths.best_model_file)
        
        results = {
            'training_time': training_time,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'vocab_size': vocab_size,
            'final_epoch': len(history['train_losses']),
            'best_model_path': best_model_path,
            'vocabulary_path': vocab_file
        }
        
        # Aggiungi metriche del training
        if history:
            results.update({
                'best_train_loss': min(history['train_losses']),
                'best_val_loss': min(history['val_losses']),
                'best_train_acc': max(history['train_accuracies']),
                'best_val_acc': max(history['val_accuracies'])
            })
        
        # Aggiungi metriche finali se disponibili
        if final_metrics:
            results.update({
                'best_val_f1': final_metrics.get('f1_macro', 0),
                'best_val_precision': final_metrics.get('precision_macro', 0),
                'best_val_recall': final_metrics.get('recall_macro', 0),
                'best_val_auc_roc': final_metrics.get('auc_roc', 0)
            })
        
        # Se esiste checkpoint con metriche dettagliate
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                if 'val_f1' in checkpoint:
                    results['best_val_f1'] = checkpoint['val_f1']
                if 'val_acc' in checkpoint:
                    results['best_val_acc'] = checkpoint['val_acc']
                if 'val_precision' in checkpoint:
                    results['best_val_precision'] = checkpoint['val_precision']
                if 'val_recall' in checkpoint:
                    results['best_val_recall'] = checkpoint['val_recall']
                if 'val_auc_roc' in checkpoint:
                    results['best_val_auc_roc'] = checkpoint['val_auc_roc']
            except Exception as e:
                logger.warning(f"Errore caricamento checkpoint per metriche: {e}")
        
        print("\n" + "="*60)
        print("ADDESTRAMENTO COMPLETATO!")
        print("="*60)
        print("File salvati:")
        print(f"- {os.path.join(models_dir, config.paths.best_model_file)}: Miglior modello")
        print(f"- {os.path.join(models_dir, config.paths.vocabulary_file)}: Vocabolario")
        if config.evaluation.plot_training_history:
            print(f"- {os.path.join(models_dir, 'training_history.png')}: Grafici addestramento")
        print(f"- {data_dir}/: Dataset processati")
        print(f"\n⏱️ Tempo totale: {training_time:.1f} secondi ({training_time/60:.1f} minuti)")
        
        logger.info(f"✅ Training completato in {training_time:.1f}s, F1-Score: {results.get('best_val_f1', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Errore durante il training: {e}")
        return {
            'error': str(e),
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }

def main():
    """
    Funzione principale per lanciare il training.
    """
    # --------------------------------------------------------------------------
    print("="*60)
    print("ADDESTRAMENTO RETE NEURALE PER SENTIMENT ANALYSIS")
    print("="*60)
    
    # Analisi memoria sistema
    memory_info = get_memory_info()
    print(f"💾 ANALISI MEMORIA SISTEMA:")
    print(f"   Memoria totale: {memory_info['total_gb']:.1f} GB")
    print(f"   Memoria disponibile: {memory_info['available_gb']:.1f} GB")
    print(f"   Memoria utilizzata: {memory_info['used_percent']:.1f}%")
    
    # Ottimizzazione automatica parametri
    optimal_params = optimize_model_for_memory(memory_info['available_gb'])
    
    # Parametri ottimizzati automaticamente
    MAX_SAMPLES = optimal_params['max_samples']
    BATCH_SIZE = optimal_params['batch_size']
    EMBEDDING_DIM = optimal_params['embedding_dim']
    HIDDEN_DIM = optimal_params['hidden_dim']
    USE_ATTENTION = optimal_params['use_attention']
    GRADIENT_ACCUMULATION_STEPS = optimal_params['gradient_accumulation_steps']
    
    # Parametri fissi
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    MAX_LENGTH = 100
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    print(f"\n🔧 PARAMETRI OTTIMIZZATI PER LA MEMORIA:")
    print(f"   Max samples: {MAX_SAMPLES:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Embedding dim: {EMBEDDING_DIM}")
    print(f"   Hidden dim: {HIDDEN_DIM}")
    print(f"   Attention: {USE_ATTENTION}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Calcola batch size ottimale teorico
    theoretical_batch_size, memory_per_sample = calculate_optimal_batch_size(
        vocab_size=10000,  # stima
        seq_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        available_memory_gb=memory_info['available_gb']
    )
    print(f"   Batch size teorico ottimale: {theoretical_batch_size}")
    print(f"   Memoria per sample: {memory_per_sample:.2f} MB")
    
    # 1. Preparazione dei dati
    print("\n1. PREPARAZIONE DEI DATI")
    print("-" * 30)
    
    # Controlla se i dati esistono già
    if os.path.exists('data/train.csv'):
        print("Caricamento dati esistenti...")
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/val.csv')
        test_df = pd.read_csv('data/test.csv')
        
        X_train, y_train = train_df['text'], train_df['sentiment']
        X_val, y_val = val_df['text'], val_df['sentiment']
        X_test, y_test = test_df['text'], test_df['sentiment']
    else:
        print("Preparazione nuovi dati...")
        preprocessor = AdvancedTweetPreprocessor()
        df = preprocessor.prepare_and_augment_dataset(augment_minority=True)
        # Limita il numero di campioni se specificato
        if MAX_SAMPLES and len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        # Usa la funzione split_data indipendente
        from data_preparation import split_data
        split_data(df)
        
        # Ricarica i dati divisi
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/val.csv')
        test_df = pd.read_csv('data/test.csv')
        
        X_train, y_train = train_df['text'], train_df['sentiment']
        X_val, y_val = val_df['text'], val_df['sentiment']
        X_test, y_test = test_df['text'], test_df['sentiment']
        
        # Salva i dati
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv('data/train.csv', index=False)
        pd.DataFrame({'text': X_val, 'sentiment': y_val}).to_csv('data/val.csv', index=False)
        pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv('data/test.csv', index=False)
    
    print(f"Dataset preparato:")
    print(f"- Training: {len(X_train)} campioni")
    print(f"- Validation: {len(X_val)} campioni")
    print(f"- Test: {len(X_test)} campioni")
    
    # 2. Costruzione del vocabolario
    print("\n2. COSTRUZIONE VOCABOLARIO")
    print("-" * 30)
    
    # Combina tutti i testi per costruire il vocabolario
    all_texts = pd.concat([X_train, X_val, X_test])
    
    # Inizializza trainer temporaneo per costruire vocabolario
    temp_model = SentimentLSTM(100, use_attention=False)  # vocab_size temporaneo, senza attention per velocità
    trainer = SentimentTrainer(temp_model, device)
    vocab_to_idx, idx_to_vocab = trainer.build_vocabulary(all_texts)
    
    # Salva il vocabolario
    save_vocabulary(vocab_to_idx, idx_to_vocab)
    
    # 3. Creazione del modello avanzato definitivo
    print("\n3. CREAZIONE MODELLO AVANZATO")
    print("-" * 30)
    
    vocab_size = len(vocab_to_idx)
    
    # Crea modello con architettura avanzata
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,        # Attention classico
        use_self_attention=True,            # Self-attention (Transformer-like)
        use_residual=True                   # Residual connections
    )
    
    trainer = SentimentTrainer(model, device)
    
    print(f"Modello AVANZATO creato:")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Embedding dim: {EMBEDDING_DIM}")
    print(f"- Hidden dim: {HIDDEN_DIM}")
    print(f"- Num layers: {NUM_LAYERS}")
    print(f"- Dropout: {DROPOUT}")
    print(f"- Attention mechanism: {USE_ATTENTION}")
    print(f"- Self-Attention (Transformer): ✅")
    print(f"- Residual Connections: ✅")
    print(f"- Positional Encoding: ✅")
    print(f"- Multi-Head Self-Attention: ✅")
    print(f"- Layer Normalization: ✅")
    print(f"- Architettura: LSTM + Self-Attention + Residual + Multi-Pooling")
    
    # Conta i parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Parametri totali: {total_params:,}")
    print(f"- Parametri addestrabili: {trainable_params:,}")
    print(f"- Memoria modello stimata: {total_params * 4 / 1024**2:.1f} MB")
    
    # 4. Creazione dataset e dataloader
    print("\n4. CREAZIONE DATALOADER")
    print("-" * 30)
    
    train_dataset = TweetDataset(X_train, y_train, vocab_to_idx, MAX_LENGTH)
    val_dataset = TweetDataset(X_val, y_val, vocab_to_idx, MAX_LENGTH)
    test_dataset = TweetDataset(X_test, y_test, vocab_to_idx, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"DataLoader creati:")
    print(f"- Train batches: {len(train_loader)}")
    print(f"- Val batches: {len(val_loader)}")
    print(f"- Test batches: {len(test_loader)}")
    
    # 5. Addestramento Avanzato
    print("\n5. ADDESTRAMENTO AVANZATO")
    print("-" * 30)
    
    os.makedirs('models', exist_ok=True)
    
    # Calcola i pesi delle classi per gestire sbilanciamento
    class_weights = calculate_class_weights('data/train.csv')
    
    # Determina se usare Focal Loss per dataset sbilanciati
    use_focal_loss = False
    if class_weights is not None:
        weight_ratio = max(class_weights) / min(class_weights)
        use_focal_loss = weight_ratio > 1.5
        print(f"📊 Rapporto sbilanciamento classi: {weight_ratio:.2f}")
        print(f"📊 Uso Focal Loss: {'✅' if use_focal_loss else '❌'}")
    
    print(f"🚀 FUNZIONALITÀ AVANZATE ATTIVATE:")
    print(f"   - Warmup Learning Rate: ✅")
    print(f"   - Cosine Annealing: ✅")
    print(f"   - Gradient Clipping: ✅")
    print(f"   - Class Weights: {'✅' if class_weights is not None else '❌'}")
    print(f"   - Focal Loss: {'✅' if use_focal_loss else '❌'}")
    print(f"   - Early Stopping Avanzato: ✅")
    print(f"   - Checkpoint Intermedi: ✅")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_focal_loss=use_focal_loss,
        class_weights=class_weights,
        use_warmup=True,
        cosine_annealing=True
    )
    
    # 6. Visualizzazione risultati
    print("\n6. VISUALIZZAZIONE RISULTATI")
    print("-" * 30)
    
    plot_training_history(history)
    
    # 7. Valutazione finale
    final_metrics = evaluate_model(trainer, test_loader, vocab_to_idx, 'models/best_model.pth')
    
    # 8. Test di esempi complessi per verificare le capacità avanzate
    print("\n8. TEST CAPACITÀ AVANZATE")
    print("-" * 30)
    
    challenging_examples = [
        "This is not bad at all, quite good actually",  # Negazione + positivo
        "I love this movie but the ending was terrible",  # Sentimenti misti
        "Absolutely fantastic amazing wonderful",  # Intensificatori multipli
        "Worst experience ever, completely disappointed",  # Negativo forte
        "Okay I guess, nothing special though",  # Neutro/negativo debole
        "Best day of my life!! So happy!!! 😊",  # Positivo forte con emoji
        "I don't hate it but it's not great either",  # Doppia negazione
        "Could be better but not the worst",  # Comparativi
    ]
    
    print("🧪 Testando capacità di comprensione avanzata...")
    
    # Preprocessing per i test
    preprocessor = AdvancedTweetPreprocessor()
    
    for i, text in enumerate(challenging_examples, 1):
        # Preprocessa il testo con il nostro sistema avanzato
        clean_text = preprocessor.clean_text(text)
        sentiment, confidence = trainer.predict(clean_text, vocab_to_idx)
        
        emoji = "😊" if sentiment == "Positivo" else "😔"
        confidence_level = "🔥" if confidence > 0.8 else "👍" if confidence > 0.6 else "🤔"
        
        print(f"{i}. {emoji} '{text}'")
        print(f"   Pulito: '{clean_text}'")
        print(f"   Predizione: {sentiment} {confidence_level} (Conf: {confidence:.4f})")
    
    print("\n" + "="*60)
    print("🚀 ADDESTRAMENTO AVANZATO COMPLETATO CON SUCCESSO!")
    print("="*60)
    print("📊 RISULTATI FINALI:")
    print(f"   🎯 Accuratezza Test: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"   🎯 F1-Score Test: {final_metrics['f1_macro']:.4f}")
    if final_metrics.get('auc_roc'):
        print(f"   🎯 AUC-ROC Test: {final_metrics['auc_roc']:.4f}")
    
    print("\n📁 FILE SALVATI:")
    print("   ✅ models/best_model.pth - Miglior modello AVANZATO")
    print("   ✅ models/vocabulary.pkl - Vocabolario del modello")
    print("   ✅ models/training_history.png - Grafici dell'addestramento")
    print("   ✅ models/final_test_metrics.txt - Metriche finali del test")
    print("   ✅ models/best_model_metrics.txt - Report dettagliato")
    print("   ✅ models/plots/ - Grafici di valutazione avanzati")
    print("   ✅ data/ - Dataset processati (train/val/test)")
    
    print("\n🎯 CARATTERISTICHE AVANZATE IMPLEMENTATE:")
    print("   ✅ Architettura ibrida LSTM + Transformer")
    print("   ✅ Multi-Head Self-Attention")
    print("   ✅ Residual Connections")
    print("   ✅ Layer Normalization")
    print("   ✅ Positional Encoding")
    print("   ✅ Preprocessing avanzato con gestione negazioni")
    print("   ✅ Data Augmentation multi-tecnica")
    print("   ✅ Focal Loss per dataset sbilanciati")
    print("   ✅ Learning Rate Scheduling avanzato")
    print("   ✅ Gradient Clipping e Weight Decay")
    
    print("\n🎯 Il modello AVANZATO è pronto per essere utilizzato!")
    print("   Puoi testarlo con: python test_model.py")
    print("   Oppure usarlo nell'app web: python app.py")

if __name__ == '__main__':
    main() 