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
from data_preparation import intelligent_truncate

class TweetDataset(Dataset):
    """Crea un dataset PyTorch per i tweet, con troncamento intelligente."""
    
    def __init__(self, texts, labels, vocab_to_idx, max_length=100, 
                 truncation_strategy='simple', head_tail_ratio=0.5):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.head_tail_ratio = head_tail_ratio
        self.pad_token_idx = self.vocab_to_idx.get('<PAD>', 0)
        self.unk_token_idx = self.vocab_to_idx.get('<UNK>', 1)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenizzazione e conversione in indici
        tokens = text.split()
        
        # Troncamento intelligente prima della conversione in indici per efficienza
        truncated_tokens = intelligent_truncate(
            tokens, 
            self.max_length, 
            self.truncation_strategy, 
            self.head_tail_ratio
        )
        
        indices = [self.vocab_to_idx.get(token, self.unk_token_idx) for token in truncated_tokens]
        
        # Padding
        if len(indices) < self.max_length:
            indices.extend([self.pad_token_idx] * (self.max_length - len(indices)))
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SentimentLSTM(nn.Module):
    """Rete neurale LSTM avanzata con Self-Attention, Residual Connections e tecniche moderne"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3, 
                 use_attention=True, use_self_attention=True, use_residual=True):
        super(SentimentLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_self_attention = use_self_attention
        self.use_residual = use_residual
        
        # Layer di embedding con inizializzazione migliorata
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Inizializzazione Xavier per embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Positional Encoding per Self-Attention
        if self.use_self_attention:
            self.pos_encoding = PositionalEncoding(embedding_dim, dropout=0.1)
        
        # LSTM layer bidirezionale
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout e Layer Normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Self-Attention layers (Transformer-like)
        if self.use_self_attention:
            self.self_attention = MultiHeadSelfAttention(
                hidden_dim * 2, num_heads=8, dropout=dropout
            )
            self.self_attn_norm = nn.LayerNorm(hidden_dim * 2)
            
            # Feed-forward network per self-attention
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim * 2)
            )
            self.ff_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Attention mechanism classico (opzionale)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Calcola la dimensione dell'input per il classificatore finale
        classifier_input_dim = hidden_dim * 2  # ultimo hidden state
        classifier_input_dim += hidden_dim * 2  # max pooling
        classifier_input_dim += hidden_dim * 2  # mean pooling
        if self.use_attention:
            classifier_input_dim += hidden_dim * 2  # attention output
        if self.use_self_attention:
            classifier_input_dim += hidden_dim * 2  # self-attention output
        
        # Dense layers con architecture più sofisticata
        self.fc = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Dropout ridotto negli ultimi layer
            
            nn.Linear(32, 2)  # 2 classi: positivo, negativo
        )
        
        # Inizializzazione pesi
        self._init_weights()
        
    def _init_weights(self):
        """Inizializzazione avanzata dei pesi"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Positional encoding per self-attention
        if self.use_self_attention:
            embedded = self.pos_encoding(embedded)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Crea mask per padding se non fornita
        if mask is None:
            mask = (x != 0).float()  # 0 è il padding token
        
        # Self-Attention con residual connection
        if self.use_self_attention:
            # Self-attention
            attn_out = self.self_attention(lstm_out, lstm_out, lstm_out, mask)
            if self.use_residual:
                attn_out = self.self_attn_norm(lstm_out + attn_out)  # Residual connection
            else:
                attn_out = self.self_attn_norm(attn_out)
            
            # Feed-forward con residual connection
            ff_out = self.feed_forward(attn_out)
            if self.use_residual:
                sequence_representation = self.ff_norm(attn_out + ff_out)  # Residual connection
            else:
                sequence_representation = self.ff_norm(ff_out)
        else:
            sequence_representation = lstm_out
        
        # 1. Ultimo hidden state (approccio originale)
        hidden_forward = hidden[-2]  # ultimo layer forward
        hidden_backward = hidden[-1]  # ultimo layer backward
        last_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # 2. Max pooling su tutta la sequenza
        masked_repr = sequence_representation * mask.unsqueeze(-1)
        max_pooled, _ = torch.max(masked_repr, dim=1)
        
        # 3. Mean pooling su tutta la sequenza
        seq_lengths = mask.sum(dim=1, keepdim=True)
        mean_pooled = masked_repr.sum(dim=1) / seq_lengths
        
        # Lista delle features da combinare
        features = [last_hidden, max_pooled, mean_pooled]
        
        # 4. Attention mechanism classico (se abilitato)
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention(sequence_representation)  # (batch_size, seq_len, 1)
            attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)
            
            # Applica mask per ignorare padding
            attention_weights = attention_weights * mask
            
            # Normalizza con softmax
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Calcola il weighted sum
            attention_output = torch.sum(sequence_representation * attention_weights.unsqueeze(-1), dim=1)
            features.append(attention_output)
        
        # 5. Self-attention pooling (se abilitato)
        if self.use_self_attention:
            # Global average pooling della rappresentazione self-attention
            self_attn_pooled = torch.mean(sequence_representation, dim=1)
            features.append(self_attn_pooled)
        
        # Combina tutte le rappresentazioni
        combined_features = torch.cat(features, dim=1)
        
        # Dropout
        dropped = self.dropout(combined_features)
        
        # Classificazione finale
        output = self.fc(dropped)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding come nei Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention come nei Transformer"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask to match attention scores shape
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(attn_output)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss per gestire meglio i dataset sbilanciati"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnsembleModel(nn.Module):
    """Ensemble di modelli per migliorare le prestazioni"""
    
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x, mask=None):
        outputs = []
        for model in self.models:
            output, _ = model(x, mask)
            outputs.append(torch.softmax(output, dim=1))
        
        # Media delle probabilità
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return torch.log(ensemble_output + 1e-8), None  # Log per compatibilità con NLLLoss

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
    
    def train_epoch(self, dataloader, optimizer, criterion, gradient_accumulation_steps=1, max_grad_norm=1.0):
        """Addestra per una epoch con supporto per gradient accumulation e clipping avanzato"""
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
                # Gradient clipping avanzato per stabilità
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Pulizia della memoria più frequente
                if batch_idx % (gradient_accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Aggiorna progress bar con più informazioni
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'Acc': f'{100 * correct / total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Mem': f'{torch.cuda.memory_allocated() / 1024**2:.0f}MB' if torch.cuda.is_available() else 'CPU'
            })
        
        # Assicurati che l'ultimo batch sia processato
        if len(dataloader) % gradient_accumulation_steps != 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
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
              model_save_path='models/best_model.pth', models_dir='models', use_focal_loss=False, 
              class_weights=None, use_warmup=True, cosine_annealing=True):
        """Allena il modello con tecniche avanzate di training"""
        
        # Crea la directory se non esiste
        os.makedirs(models_dir, exist_ok=True)
        
        # Ottimizzatore avanzato
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Criterio di perdita avanzato
        if use_focal_loss:
            criterion = FocalLoss(alpha=1, gamma=2)
            print("📊 Usando Focal Loss per dataset sbilanciati")
        elif class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            print(f"📊 Usando CrossEntropyLoss con pesi delle classi: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("📊 Usando CrossEntropyLoss standard")
        
        # Learning rate scheduling avanzato
        if use_warmup:
            warmup_epochs = max(1, num_epochs // 10)  # 10% del training per warmup
            print(f"🔥 Warmup attivato per {warmup_epochs} epochs")
        
        if cosine_annealing:
            # Cosine Annealing con restart
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(5, num_epochs // 4), eta_min=learning_rate * 0.01
            )
        else:
            # ReduceLROnPlateau tradizionale
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        
        # Early stopping avanzato
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_auc = 0.0
        patience = max(7, num_epochs // 5)  # Patience adattiva
        patience_counter = 0
        no_improvement_threshold = 0.001  # Miglioramento minimo richiesto
        
        # Gradient clipping
        max_grad_norm = 1.0
        
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'val_f1_scores': [],
            'learning_rates': []
        }
        
        print(f"🚀 Inizio training avanzato per {num_epochs} epochs...")
        print(f"📊 Parametri avanzati:")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - Batch Accumulation: {gradient_accumulation_steps}")
        print(f"   - Patience: {patience}")
        print(f"   - Warmup: {use_warmup}")
        print(f"   - Cosine Annealing: {cosine_annealing}")
        print(f"   - Gradient Clipping: {max_grad_norm}")
        
        import time
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"🔄 EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Warmup learning rate
            if use_warmup and epoch < warmup_epochs:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"🔥 Warmup LR: {warmup_lr:.6f}")
            
            # Training con gradient clipping
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, gradient_accumulation_steps, max_grad_norm
            )
            
            # Validazione con metriche dettagliate
            val_loss, val_detailed_metrics, val_metrics_obj = self.validate(
                val_loader, criterion, compute_detailed_metrics=True
            )
            
            # Estrai metriche chiave
            val_acc = val_detailed_metrics['accuracy']
            val_f1 = val_detailed_metrics['f1_macro']
            val_auc = val_detailed_metrics.get('auc_roc', 0.0)
            
            # Aggiorna scheduler
            if cosine_annealing:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Salva nella history
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_accuracies'].append(train_acc)
            history['val_accuracies'].append(val_acc)
            history['val_f1_scores'].append(val_f1)
            history['learning_rates'].append(current_lr)
            
            # Print dei risultati
            print(f"\n📈 RISULTATI EPOCH {epoch+1}:")
            print(f"   🔹 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"   🔹 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"   🔹 Val F1-Score: {val_f1:.4f} | Val AUC-ROC: {val_auc:.4f}")
            print(f"   🔹 Learning Rate: {current_lr:.6f}")
            print(f"   🔹 Memoria: {get_memory_usage()}")
            
            # Early stopping migliorato con soglia di miglioramento
            improved = False
            improvement_score = val_f1 + 0.3 * val_acc + 0.2 * val_auc  # Score composito
            
            if epoch == 0:
                best_improvement_score = improvement_score
                improved = True
            else:
                if improvement_score > best_improvement_score + no_improvement_threshold:
                    improved = True
                    best_improvement_score = improvement_score
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if improved:
                # Salva il modello migliore
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_auc_roc': val_auc,
                    'history': history,
                    'detailed_metrics': val_detailed_metrics,
                    'improvement_score': improvement_score
                }
                
                torch.save(checkpoint, model_save_path)
                print(f"   ✅ Nuovo miglior modello salvato! Score: {improvement_score:.4f}")
                
                # Salva anche le metriche dettagliate
                val_metrics_obj.save_metrics_report(
                    f'{models_dir}/best_model_metrics.txt', 
                    f"Best Model - Epoch {epoch+1}"
                )
            else:
                print(f"   ⏳ Nessun miglioramento significativo. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING! Nessun miglioramento per {patience} epochs consecutive.")
                break
            
            # Report dettagliato ogni 10 epoche
            if (epoch + 1) % 10 == 0:
                print(f"\n📊 REPORT DETTAGLIATO EPOCH {epoch+1}:")
                val_metrics_obj.print_detailed_report(f"Validation Epoch {epoch+1}")
            
            # Salvataggio checkpoint intermedio
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f'{models_dir}/checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, checkpoint_path)
                print(f"   💾 Checkpoint salvato: {checkpoint_path}")
            
            # Gestione memoria
            clear_memory()
            
            # Stima tempo rimanente
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            
            print(f"   ⏱️ Tempo: {elapsed_time/60:.1f}m | Stimato rimanente: {estimated_remaining/60:.1f}m")
        
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETATO!")
        print(f"⏱️ Tempo totale: {total_time/60:.2f} minuti")
        print(f"🏆 Migliori metriche raggiunte:")
        print(f"   - Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   - Val F1-Score: {best_val_f1:.4f}")
        print(f"   - Val AUC-ROC: {best_val_auc:.4f}")
        
        return history
    
    def predict(self, text, vocab_to_idx, max_length=100, 
              truncation_strategy='simple', head_tail_ratio=0.5):
        """Predice il sentiment di un singolo testo con troncamento intelligente"""
        self.model.eval()
        
        # Preprocessa il testo
        tokens = text.split()
        
        # Troncamento intelligente
        truncated_tokens = intelligent_truncate(
            tokens, 
            max_length, 
            truncation_strategy, 
            head_tail_ratio
        )
        
        indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in truncated_tokens]
        
        # Padding o troncamento finale (di sicurezza)
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([vocab_to_idx.get('<PAD>', 0)] * (max_length - len(indices)))
        
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

def get_memory_info_complete():
    """Ottiene informazioni complete sulla memoria (unifica tutte le implementazioni)"""
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_info = process.memory_info()
    
    result = {
        # Informazioni sistema (da train_sentiment_model.py e test_model.py)
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'free_gb': memory.free / (1024**3),
        'used_percent': memory.percent,
        
        # Informazioni processo corrente
        'process_mb': process_info.rss / (1024 * 1024),
        
        # Informazioni GPU se disponibile
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory_mb': 0
    }
    
    if torch.cuda.is_available():
        result['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return result

def get_memory_usage():
    """Ottiene l'uso della memoria corrente (formato stringa per compatibilità)"""
    info = get_memory_info_complete()
    
    if info['gpu_available']:
        return f"RAM: {info['process_mb']:.0f}MB, GPU: {info['gpu_memory_mb']:.0f}MB"
    else:
        return f"RAM: {info['process_mb']:.0f}MB"

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