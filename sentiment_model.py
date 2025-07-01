import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

class TweetDataset(Dataset):
    """Dataset personalizzato per i tweet"""
    
    def __init__(self, texts, labels, vocab_to_idx, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Converti testo in indici
        tokens = text.split()
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Padding o troncamento
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SentimentLSTM(nn.Module):
    """Rete neurale LSTM per sentiment analysis"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layer di embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer di classificazione finale
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 per LSTM bidirezionale
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classi: positivo, negativo
        )
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Prendi l'ultimo output (last time step)
        # Per LSTM bidirezionale, concateniamo gli hidden states forward e backward
        hidden_forward = hidden[-2]  # ultimo layer forward
        hidden_backward = hidden[-1]  # ultimo layer backward
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Dropout
        dropped = self.dropout(hidden_concat)
        
        # Classificazione finale
        output = self.fc(dropped)
        
        return output

class SentimentTrainer:
    """Classe per l'addestramento del modello"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def build_vocabulary(self, texts, min_freq=2, max_vocab_size=10000):
        """Costruisce il vocabolario dal testo"""
        
        # Conta tutte le parole
        word_counts = Counter()
        for text in texts:
            if pd.isna(text):
                continue
            words = str(text).split()
            word_counts.update(words)
        
        # Seleziona le parole più frequenti
        most_common = word_counts.most_common(max_vocab_size - 3)  # -3 per <PAD>, <UNK>, <START>
        
        # Filtra per frequenza minima
        vocab_words = [word for word, count in most_common if count >= min_freq]
        
        # Crea il mapping
        vocab_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2
        }
        
        for i, word in enumerate(vocab_words):
            vocab_to_idx[word] = i + 3
        
        idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
        
        print(f"Vocabolario costruito: {len(vocab_to_idx)} parole")
        print(f"Parole più comuni: {list(vocab_to_idx.keys())[3:13]}")
        
        return vocab_to_idx, idx_to_vocab
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Addestra per una epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_texts, batch_labels in progress_bar:
            batch_texts = batch_texts.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistiche
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Aggiorna progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader, criterion):
        """Valuta il modello"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Addestra il modello completo"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25, verbose=True)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = 50  # Stop se non migliora per 50 epoche
        
        print("Inizio addestramento...")
        print(f"Early stopping attivo: si fermerà se non migliora per {early_stopping_patience} epoche")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Addestramento
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validazione
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Salva le metriche
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Salva il miglior modello
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, 'models/best_model.pth')
                print(f"🎉 Nuovo miglior modello salvato! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"Nessun miglioramento. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Salva checkpoint ogni 25 epoche
            if (epoch + 1) % 25 == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, f'models/checkpoint_epoch_{epoch+1}.pth')
                print(f"💾 Checkpoint salvato per epoch {epoch+1}")
            
            # Aggiorna learning rate
            scheduler.step(val_acc)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping attivato dopo {epoch+1} epoche")
                print(f"Miglior validation accuracy: {best_val_acc:.4f}")
                break
                
            # Progress update ogni 10 epoche
            if (epoch + 1) % 10 == 0:
                print(f"\n📊 Progress Update - Epoch {epoch+1}:")
                print(f"   Miglior Val Acc: {best_val_acc:.4f}")
                print(f"   Current Val Acc: {val_acc:.4f}")
                print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"\n✅ Addestramento completato!")
        print(f"   Epoche effettuate: {len(train_losses)}")
        print(f"   Miglior accuracy: {best_val_acc:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def predict(self, text, vocab_to_idx, max_length=100):
        """Predice il sentiment di un singolo testo"""
        self.model.eval()
        
        # Preprocessa il testo
        tokens = text.split()
        indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
        
        # Padding o troncamento
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([vocab_to_idx['<PAD>']] * (max_length - len(indices)))
        
        # Converti in tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = "Positivo" if predicted_class == 1 else "Negativo"
        return sentiment, confidence

def save_vocabulary(vocab_to_idx, idx_to_vocab, filename='models/vocabulary.pkl'):
    """Salva il vocabolario"""
    os.makedirs('models', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'vocab_to_idx': vocab_to_idx, 'idx_to_vocab': idx_to_vocab}, f)

def load_vocabulary(filename='models/vocabulary.pkl'):
    """Carica il vocabolario"""
    with open(filename, 'rb') as f:
        vocab_data = pickle.load(f)
    return vocab_data['vocab_to_idx'], vocab_data['idx_to_vocab'] 