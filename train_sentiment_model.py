import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import TweetPreprocessor
from sentiment_model import SentimentLSTM, SentimentTrainer, TweetDataset, save_vocabulary
import os

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

def evaluate_model(trainer, test_loader, vocab_to_idx):
    """Valuta il modello sul test set"""
    print("\n" + "="*50)
    print("VALUTAZIONE FINALE SUL TEST SET")
    print("="*50)
    
    # Carica il miglior modello
    checkpoint = torch.load('models/best_model.pth')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Valuta sul test set
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.validate(test_loader, criterion)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Test su esempi specifici
    print("\n" + "="*30)
    print("TEST SU ESEMPI SPECIFICI")
    print("="*30)
    
    test_texts = [
        "I love this beautiful sunny day",
        "This is the worst movie I have ever seen",
        "Amazing experience highly recommend",
        "Terrible service very disappointed",
        "Perfect weather for walking outside",
        "Feeling sad and frustrated today"
    ]
    
    for text in test_texts:
        # Preprocessing del testo (simile a quello fatto nel dataset)
        from data_preparation import TweetPreprocessor
        preprocessor = TweetPreprocessor()
        clean_text = preprocessor.clean_text(text)
        
        sentiment, confidence = trainer.predict(clean_text, vocab_to_idx)
        print(f"Testo: '{text}'")
        print(f"Testo pulito: '{clean_text}'")
        print(f"Predizione: {sentiment} (Confidence: {confidence:.4f})")
        print("-" * 40)

def main():
    """Funzione principale per l'addestramento completo"""
    
    print("="*60)
    print("ADDESTRAMENTO RETE NEURALE PER SENTIMENT ANALYSIS")
    print("="*60)
    
    # Parametri
    MAX_SAMPLES = 20000
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    MAX_LENGTH = 100
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
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
        preprocessor = TweetPreprocessor()
        df = preprocessor.prepare_dataset(max_samples=MAX_SAMPLES)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
        
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
    temp_model = SentimentLSTM(100)  # vocab_size temporaneo
    trainer = SentimentTrainer(temp_model, device)
    vocab_to_idx, idx_to_vocab = trainer.build_vocabulary(all_texts)
    
    # Salva il vocabolario
    save_vocabulary(vocab_to_idx, idx_to_vocab)
    
    # 3. Creazione del modello definitivo
    print("\n3. CREAZIONE MODELLO")
    print("-" * 30)
    
    vocab_size = len(vocab_to_idx)
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    trainer = SentimentTrainer(model, device)
    
    print(f"Modello creato:")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Embedding dim: {EMBEDDING_DIM}")
    print(f"- Hidden dim: {HIDDEN_DIM}")
    print(f"- Num layers: {NUM_LAYERS}")
    print(f"- Dropout: {DROPOUT}")
    
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
    
    os.makedirs('models', exist_ok=True)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # 6. Visualizzazione risultati
    print("\n6. VISUALIZZAZIONE RISULTATI")
    print("-" * 30)
    
    plot_training_history(history)
    
    # 7. Valutazione finale
    evaluate_model(trainer, test_loader, vocab_to_idx)
    
    print("\n" + "="*60)
    print("ADDESTRAMENTO COMPLETATO!")
    print("="*60)
    print("File salvati:")
    print("- models/best_model.pth: Miglior modello")
    print("- models/vocabulary.pkl: Vocabolario")
    print("- models/training_history.png: Grafici addestramento")
    print("- data/: Dataset processati")

if __name__ == "__main__":
    main() 