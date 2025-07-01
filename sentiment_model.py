import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import gc
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Rete neurale LSTM migliorata per sentiment analysis con attention e pooling"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3, use_attention=True):
        super(SentimentLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
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
        
        # Attention mechanism (opzionale)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Calcola la dimensione dell'input per il classificatore finale
        # Combiniamo: attention output + max pooling + mean pooling + ultimo hidden state
        classifier_input_dim = hidden_dim * 2  # ultimo hidden state
        classifier_input_dim += hidden_dim * 2  # max pooling
        classifier_input_dim += hidden_dim * 2  # mean pooling
        if self.use_attention:
            classifier_input_dim += hidden_dim * 2  # attention output
        
        # Layer di classificazione finale con più informazioni
        self.fc = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classi: positivo, negativo
        )
        
    def forward(self, x, mask=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Crea mask per padding se non fornita
        if mask is None:
            mask = (x != 0).float()  # 0 è il padding token
        
        # 1. Ultimo hidden state (approccio originale)
        hidden_forward = hidden[-2]  # ultimo layer forward
        hidden_backward = hidden[-1]  # ultimo layer backward
        last_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # 2. Max pooling su tutta la sequenza
        # Mascheriamo i token di padding prima del pooling
        masked_lstm_out = lstm_out * mask.unsqueeze(-1)
        max_pooled, _ = torch.max(masked_lstm_out, dim=1)
        
        # 3. Mean pooling su tutta la sequenza
        # Calcoliamo la media solo sui token non-padding
        seq_lengths = mask.sum(dim=1, keepdim=True)
        mean_pooled = masked_lstm_out.sum(dim=1) / seq_lengths
        
        # 4. Attention mechanism (se abilitato)
        features = [last_hidden, max_pooled, mean_pooled]
        
        if self.use_attention:
            # Calcola attention weights
            attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)
            
            # Applica mask per ignorare padding
            attention_weights = attention_weights * mask
            
            # Normalizza con softmax
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Calcola il weighted sum
            attention_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
            features.append(attention_output)
        
        # Combina tutte le rappresentazioni
        combined_features = torch.cat(features, dim=1)
        
        # Dropout
        dropped = self.dropout(combined_features)
        
        # Classificazione finale
        output = self.fc(dropped)
        
        return output, attention_weights if self.use_attention else None

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
    
    def train_epoch(self, dataloader, optimizer, criterion, gradient_accumulation_steps=1):
        """Addestra per una epoch con supporto per gradient accumulation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        # Reset gradients all'inizio
        optimizer.zero_grad()
        
        for batch_idx, (batch_texts, batch_labels) in enumerate(progress_bar):
            batch_texts = batch_texts.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            outputs, _ = self.model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            # Scala la loss per gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Statistiche (usa loss non scalata per reporting)
            total_loss += loss.item() * gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Update weights ogni gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping per stabilità
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Pulizia della memoria
                if batch_idx % (gradient_accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Aggiorna progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'Acc': f'{100 * correct / total:.2f}%',
                'Mem': f'{torch.cuda.memory_allocated() / 1024**2:.0f}MB' if torch.cuda.is_available() else 'CPU'
            })
        
        # Assicurati che l'ultimo batch sia processato
        if len(dataloader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader, criterion, compute_detailed_metrics=True):
        """Valuta il modello con metriche complete"""
        self.model.eval()
        total_loss = 0
        metrics = SentimentMetrics() if compute_detailed_metrics else None
        
        with torch.no_grad():
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs, _ = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                
                if compute_detailed_metrics:
                    # Calcola probabilità e predizioni
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Aggiorna metriche (prendi probabilità della classe positiva)
                    y_scores = probabilities[:, 1] if probabilities.size(1) > 1 else probabilities[:, 0]
                    metrics.update(batch_labels, predicted, y_scores)
        
        avg_loss = total_loss / len(dataloader)
        
        if compute_detailed_metrics:
            detailed_metrics = metrics.compute_metrics()
            return avg_loss, detailed_metrics, metrics
        else:
            # Backward compatibility
            correct = sum(metrics.y_true == metrics.y_pred) if metrics else 0
            total = len(metrics.y_true) if metrics else 0
            accuracy = correct / total if total > 0 else 0
            return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001, gradient_accumulation_steps=1, 
              model_save_path='models/best_model.pth', models_dir='models'):
        """Addestra il modello completo"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25)
        
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
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, gradient_accumulation_steps)
            
            # Validazione con metriche dettagliate
            val_loss, val_detailed_metrics, val_metrics_obj = self.validate(val_loader, criterion, compute_detailed_metrics=True)
            val_acc = val_detailed_metrics['accuracy']
            val_f1 = val_detailed_metrics['f1_macro']
            
            # Salva le metriche
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val F1-Score: {val_f1:.4f}, Val Precision: {val_detailed_metrics['precision_macro']:.4f}")
            print(f"Val Recall: {val_detailed_metrics['recall_macro']:.4f}")
            if val_detailed_metrics['auc_roc'] is not None:
                print(f"Val AUC-ROC: {val_detailed_metrics['auc_roc']:.4f}")
            print(f"Memoria: {get_memory_usage()}")
            
            # Salva il miglior modello (ora basato su F1-score invece che accuracy)
            current_metric = val_f1  # Usa F1-score come metrica principale
            if current_metric > best_val_acc:
                best_val_acc = current_metric
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_precision': val_detailed_metrics['precision_macro'],
                    'val_recall': val_detailed_metrics['recall_macro'],
                    'val_auc_roc': val_detailed_metrics['auc_roc'],
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'detailed_metrics': val_detailed_metrics
                }, model_save_path)
                print(f"🎉 Nuovo miglior modello salvato! Val F1: {val_f1:.4f}")
                
                # Salva anche report dettagliato del miglior modello
                val_metrics_obj.save_metrics_report(
                    os.path.join(models_dir, f'best_model_metrics_epoch_{epoch+1}.txt'), 
                    phase=f"Validation Epoch {epoch+1}"
                )
            else:
                patience_counter += 1
                print(f"Nessun miglioramento. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Report dettagliato ogni 10 epoche
            if (epoch + 1) % 10 == 0:
                print(f"\n📊 REPORT DETTAGLIATO EPOCH {epoch+1}:")
                val_metrics_obj.print_detailed_report(f"Validation Epoch {epoch+1}")
            
            # Salva checkpoint ogni 25 epoche
            if (epoch + 1) % 25 == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"💾 Checkpoint salvato per epoch {epoch+1}")
                clear_memory()  # Pulizia dopo checkpoint
            
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
            output, _ = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = "Positivo" if predicted_class == 1 else "Negativo"
        return sentiment, confidence

def save_vocabulary(vocab_to_idx, idx_to_vocab, filename='models/vocabulary.pkl'):
    """Salva il vocabolario"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump({'vocab_to_idx': vocab_to_idx, 'idx_to_vocab': idx_to_vocab}, f)

def load_vocabulary(filename='models/vocabulary.pkl'):
    """Carica il vocabolario"""
    with open(filename, 'rb') as f:
        vocab_data = pickle.load(f)
    return vocab_data['vocab_to_idx'], vocab_data['idx_to_vocab']

def clear_memory():
    """Pulisce la memoria per prevenire memory leaks"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_usage():
    """Ottiene l'uso della memoria corrente"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        return f"RAM: {memory_usage_mb:.0f}MB, GPU: {gpu_memory_mb:.0f}MB"
    else:
        return f"RAM: {memory_usage_mb:.0f}MB"

class SentimentMetrics:
    """Classe per calcolare e visualizzare metriche avanzate di sentiment analysis"""
    
    def __init__(self, class_names=['Negativo', 'Positivo']):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset delle metriche per una nuova epoch"""
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def update(self, y_true, y_pred, y_scores=None):
        """Aggiorna le metriche con nuovi batch"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.cpu().numpy()
            
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if y_scores is not None:
            self.y_scores.extend(y_scores)
    
    def compute_metrics(self):
        """Calcola tutte le metriche"""
        if len(self.y_true) == 0:
            return {}
            
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_scores = np.array(self.y_scores) if self.y_scores else None
        
        # Metriche di base
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Metriche per classe
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Matrice di confusione
        cm = confusion_matrix(y_true, y_pred)
        
        # Metriche ROC se abbiamo le probabilità
        auc_roc = None
        auc_pr = None
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                auc_pr = average_precision_score(y_true, y_scores)
            except ValueError:
                pass
        
        # Calcoli aggiuntivi dalla confusion matrix
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # stesso del recall per classe positiva
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
    
    def print_detailed_report(self, phase="Validation"):
        """Stampa un report dettagliato delle metriche"""
        metrics = self.compute_metrics()
        
        print(f"\n{'='*50}")
        print(f"📊 REPORT DETTAGLIATO - {phase.upper()}")
        print(f"{'='*50}")
        
        # Metriche principali
        print(f"🎯 METRICHE PRINCIPALI:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"   Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        if metrics['auc_roc'] is not None:
            print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        if metrics['auc_pr'] is not None:
            print(f"   AUC-PR: {metrics['auc_pr']:.4f}")
        
        # Metriche per classe
        print(f"\n📋 METRICHE PER CLASSE:")
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['precision_per_class']):
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision_per_class'][i]:.4f}")
                print(f"     Recall: {metrics['recall_per_class'][i]:.4f}")
                print(f"     F1-Score: {metrics['f1_per_class'][i]:.4f}")
        
        # Metriche mediche/statistiche
        print(f"\n🔬 METRICHE STATISTICHE:")
        print(f"   Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
        print(f"   Specificity (True Negative Rate): {metrics['specificity']:.4f}")
        
        # Confusion Matrix dettagli
        print(f"\n🎲 CONFUSION MATRIX:")
        print(f"   True Positives: {metrics['true_positives']}")
        print(f"   True Negatives: {metrics['true_negatives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None, figsize=(8, 6)):
        """Visualizza la confusion matrix"""
        metrics = self.compute_metrics()
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, save_path=None, figsize=(8, 6)):
        """Visualizza la ROC curve"""
        if not self.y_scores:
            print("❌ Impossibile plottare ROC curve: probabilità non disponibili")
            return
            
        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        
        if len(np.unique(y_true)) < 2:
            print("❌ Impossibile plottare ROC curve: serve almeno una classe per tipo")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None, figsize=(8, 6)):
        """Visualizza la Precision-Recall curve"""
        if not self.y_scores:
            print("❌ Impossibile plottare PR curve: probabilità non disponibili")
            return
            
        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        
        if len(np.unique(y_true)) < 2:
            print("❌ Impossibile plottare PR curve: serve almeno una classe per tipo")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {auc_pr:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_metrics_report(self, filepath, phase="Validation"):
        """Salva un report dettagliato in un file"""
        metrics = self.compute_metrics()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SENTIMENT ANALYSIS - REPORT DETTAGLIATO\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Data: {pd.Timestamp.now()}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"METRICHE PRINCIPALI:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n")
            
            if metrics['auc_roc'] is not None:
                f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
            if metrics['auc_pr'] is not None:
                f.write(f"AUC-PR: {metrics['auc_pr']:.4f}\n")
            
            f.write(f"\nMETRICHE PER CLASSE:\n")
            for i, class_name in enumerate(self.class_names):
                if i < len(metrics['precision_per_class']):
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                    f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"True Positives: {metrics['true_positives']}\n")
            f.write(f"True Negatives: {metrics['true_negatives']}\n")
            f.write(f"False Positives: {metrics['false_positives']}\n")
            f.write(f"False Negatives: {metrics['false_negatives']}\n")
        
        print(f"📄 Report salvato in: {filepath}") 