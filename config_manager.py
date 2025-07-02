"""
Configuration management system for Tweet Sentiment Analysis
Supports YAML configurations, hyperparameter tuning and validation
"""

import yaml
import os
import logging
import json
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import psutil
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProjectConfig:
    """Project configuration settings"""
    name: str = "tweet-sentiment-analysis"
    version: str = "2.0.0" 
    description: str = "Advanced sentiment analysis system"
    author: str = "AI Assistant"

@dataclass
class DataConfig:
    """Dataset configuration settings"""
    max_samples: int = 20000
    max_length: int = 100
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    min_vocab_freq: int = 2
    vocab_size_limit: int = 20000
    balance_dataset: bool = True
    positive_ratio: float = 0.5
    truncation_strategy: str = 'simple'
    head_tail_ratio: float = 0.5
    
    # Preprocessing
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags: bool = False
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    min_text_length: int = 3

@dataclass
class ModelConfig:
    """Model architecture configuration settings"""
    embedding_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    use_self_attention: bool = False
    use_residual: bool = False
    use_batch_norm: bool = True
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Classifier
    classifier_hidden_layers: List[int] = None
    classifier_activation: str = "relu"
    classifier_use_batch_norm: bool = True
    classifier_final_dropout: float = 0.3
    
    # Pooling
    use_max_pooling: bool = True
    use_mean_pooling: bool = True
    use_last_hidden: bool = True
    use_attention_pooling: bool = True
    
    def __post_init__(self):
        if self.classifier_hidden_layers is None:
            self.classifier_hidden_layers = [128, 64, 32]

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    num_epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    optimizer_type: str = "adam"
    adam_betas: List[float] = None
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    
    # Scheduler
    scheduler_type: str = "reduce_on_plateau"
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 25
    scheduler_min_lr: float = 1e-6
    cosine_T_max: int = 100
    cosine_eta_min: float = 1e-6
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 50
    early_stopping_metric: str = "f1_score"
    early_stopping_min_delta: float = 0.0001
    
    # Checkpointing
    save_best: bool = True
    save_every_n_epochs: int = 25
    keep_last_n: int = 3
    
    def __post_init__(self):
        if self.adam_betas is None:
            self.adam_betas = [0.9, 0.999]

@dataclass
class HardwareConfig:
    """Hardware and memory configuration settings"""
    auto_memory_management: bool = True
    memory_threshold: float = 0.8
    dynamic_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 64
    mixed_precision: bool = False
    dataloader_workers: int = 4
    pin_memory: bool = True
    cleanup_every_n_batches: int = 50
    force_gc_every_epoch: bool = True

@dataclass
class EvaluationConfig:
    """Metrics and evaluation configuration settings"""
    primary_metric: str = "f1_score"
    
    # Metrics
    compute_accuracy: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    compute_f1_score: bool = True
    compute_auc_roc: bool = True
    compute_auc_pr: bool = True
    compute_confusion_matrix: bool = True
    compute_sensitivity: bool = True
    compute_specificity: bool = True
    
    # Plots
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall_curve: bool = True
    plot_training_history: bool = True
    plot_attention_visualization: bool = True
    
    # Reports
    save_per_epoch: bool = False
    save_best_model: bool = True
    save_final_test: bool = True

@dataclass
class HyperparameterTuningConfig:
    """Hyperparameter tuning configuration settings"""
    enabled: bool = False
    method: str = "optuna"  # optuna, grid_search, random_search
    n_trials: int = 100
    timeout: int = 7200  # 2 hours
    
    # Search spaces
    embedding_dims: List[int] = None
    hidden_dims: List[int] = None
    num_layers_options: List[int] = None
    dropout_values: List[float] = None
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    optimizer_types: List[str] = None
    weight_decays: List[float] = None
    use_attention_options: List[bool] = None
    classifier_architectures: List[List[int]] = None
    
    # Grid search specific
    grid_search_params: Dict[str, List] = None
    
    # Random search
    n_random_samples: int = 50
    
    def __post_init__(self):
        if self.embedding_dims is None:
            self.embedding_dims = [64, 128, 256]
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]
        if self.num_layers_options is None:
            self.num_layers_options = [1, 2, 3]
        if self.dropout_values is None:
            self.dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        if self.learning_rates is None:
            self.learning_rates = [0.0001, 0.001, 0.01]
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64]
        if self.optimizer_types is None:
            self.optimizer_types = ["adam", "adamw"]
        if self.weight_decays is None:
            self.weight_decays = [1e-6, 1e-5, 1e-4]
        if self.use_attention_options is None:
            self.use_attention_options = [True, False]
        if self.classifier_architectures is None:
            self.classifier_architectures = [[64, 32], [128, 64, 32], [256, 128, 64]]
        if self.grid_search_params is None:
            self.grid_search_params = {
                'embedding_dim': [64, 128],
                'hidden_dim': [32, 64],
                'learning_rate': [0.001, 0.01],
                'dropout': [0.2, 0.3]
            }

@dataclass
class PathsConfig:
    """File paths configuration settings"""
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    results_dir: str = "results"
    vocabulary_file: str = "vocabulary.pkl"
    best_model_file: str = "best_model.pth"
    config_file: str = "config.yaml"
    tuning_results_dir: str = "hyperparameter_tuning"
    study_file: str = "optuna_study.db"

class ConfigManager:
    """Centralized configuration manager with hyperparameter tuning support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config_data = {}
        
        # Carica configurazione
        self.load_config()
        
        # Crea oggetti di configurazione
        self.project = self._create_project_config()
        self.data = self._create_data_config()
        self.model = self._create_model_config()
        self.training = self._create_training_config()
        self.hardware = self._create_hardware_config()
        self.evaluation = self._create_evaluation_config()
        self.hyperparameter_tuning = self._create_hyperparameter_tuning_config()
        self.paths = self._create_paths_config()
        
        # Auto-ottimizzazione basata su hardware
        if self.hardware.auto_memory_management:
            self._auto_optimize_for_hardware()
        
        # Setup riproducibilità
        self._setup_reproducibility()
        
        # Crea directory necessarie
        self._create_directories()
        
        logger.info(f"✅ ConfigManager inizializzato con {self.config_path}")
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
                logger.info(f"📁 Configurazione caricata da {self.config_path}")
            except Exception as e:
                logger.warning(f"Errore caricamento config: {e}, uso configurazione di default")
                self.config_data = {}
        else:
            logger.info("File config non trovato, uso configurazione di default")
            self.config_data = {}
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        # Create complete structure
        full_config = {
            'project': asdict(self.project),
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'hardware': asdict(self.hardware),
            'evaluation': asdict(self.evaluation),
            'hyperparameter_tuning': asdict(self.hyperparameter_tuning),
            'paths': asdict(self.paths)
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_config, f, default_flow_style=False, indent=2)
            logger.info(f"💾 Configurazione salvata in {save_path}")
        except Exception as e:
            logger.error(f"Errore salvataggio config: {e}")
    
    def _create_project_config(self) -> ProjectConfig:
        """Create project configuration"""
        project_data = self.config_data.get('project', {})
        return ProjectConfig(**project_data)
    
    def _create_data_config(self) -> DataConfig:
        """Create data configuration"""
        data_config = self.config_data.get('data', {})
        preprocessing = data_config.get('preprocessing', {})
        
        # Merge preprocessing config
        flat_data_config = {**data_config}
        flat_data_config.update(preprocessing)
        flat_data_config.pop('preprocessing', None)
        
        return DataConfig(**flat_data_config)
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration"""
        model_data = self.config_data.get('model', {})
        classifier_data = model_data.get('classifier', {})
        pooling_data = model_data.get('pooling', {})
        
        # Flatten nested config
        flat_config = {**model_data}
        
        # Classifier
        flat_config['classifier_hidden_layers'] = classifier_data.get('hidden_layers', [128, 64, 32])
        flat_config['classifier_activation'] = classifier_data.get('activation', 'relu')
        flat_config['classifier_use_batch_norm'] = classifier_data.get('use_batch_norm', True)
        flat_config['classifier_final_dropout'] = classifier_data.get('final_dropout', 0.3)
        
        # Pooling
        flat_config['use_max_pooling'] = pooling_data.get('use_max_pooling', True)
        flat_config['use_mean_pooling'] = pooling_data.get('use_mean_pooling', True)
        flat_config['use_last_hidden'] = pooling_data.get('use_last_hidden', True)
        flat_config['use_attention_pooling'] = pooling_data.get('use_attention_pooling', True)
        
        # Remove nested sections
        flat_config.pop('classifier', None)
        flat_config.pop('pooling', None)
        
        return ModelConfig(**flat_config)
    
    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration"""
        training_data = self.config_data.get('training', {})
        optimizer_data = training_data.get('optimizer', {})
        scheduler_data = training_data.get('scheduler', {})
        early_stopping_data = training_data.get('early_stopping', {})
        checkpointing_data = training_data.get('checkpointing', {})
        
        # Flatten config
        flat_config = {**training_data}
        
        # Optimizer
        flat_config['optimizer_type'] = optimizer_data.get('type', 'adam')
        adam_data = optimizer_data.get('adam', {})
        flat_config['adam_betas'] = adam_data.get('betas', [0.9, 0.999])
        flat_config['adam_eps'] = adam_data.get('eps', 1e-8)
        sgd_data = optimizer_data.get('sgd', {})
        flat_config['sgd_momentum'] = sgd_data.get('momentum', 0.9)
        flat_config['sgd_nesterov'] = sgd_data.get('nesterov', True)
        
        # Scheduler
        flat_config['scheduler_type'] = scheduler_data.get('type', 'reduce_on_plateau')
        plateau_data = scheduler_data.get('reduce_on_plateau', {})
        flat_config['scheduler_mode'] = plateau_data.get('mode', 'max')
        flat_config['scheduler_factor'] = plateau_data.get('factor', 0.5)
        flat_config['scheduler_patience'] = plateau_data.get('patience', 25)
        flat_config['scheduler_min_lr'] = plateau_data.get('min_lr', 1e-6)
        cosine_data = scheduler_data.get('cosine_annealing', {})
        flat_config['cosine_T_max'] = cosine_data.get('T_max', 100)
        flat_config['cosine_eta_min'] = cosine_data.get('eta_min', 1e-6)
        
        # Early stopping
        flat_config['early_stopping_enabled'] = early_stopping_data.get('enabled', True)
        flat_config['early_stopping_patience'] = early_stopping_data.get('patience', 50)
        flat_config['early_stopping_metric'] = early_stopping_data.get('metric', 'f1_score')
        flat_config['early_stopping_min_delta'] = early_stopping_data.get('min_delta', 0.0001)
        
        # Checkpointing
        flat_config['save_best'] = checkpointing_data.get('save_best', True)
        flat_config['save_every_n_epochs'] = checkpointing_data.get('save_every_n_epochs', 25)
        flat_config['keep_last_n'] = checkpointing_data.get('keep_last_n', 3)
        
        # Remove nested sections
        for key in ['optimizer', 'scheduler', 'early_stopping', 'checkpointing']:
            flat_config.pop(key, None)
        
        return TrainingConfig(**flat_config)
    
    def _create_hardware_config(self) -> HardwareConfig:
        """Create hardware configuration"""
        hardware_data = self.config_data.get('hardware', {})
        return HardwareConfig(**hardware_data)
    
    def _create_evaluation_config(self) -> EvaluationConfig:
        """Create evaluation configuration"""
        eval_data = self.config_data.get('evaluation', {})
        metrics_data = eval_data.get('metrics', {})
        plots_data = eval_data.get('plots', {})
        reports_data = eval_data.get('detailed_reports', {})
        
        # Flatten config
        flat_config = {**eval_data}
        
        # Metrics
        flat_config['compute_accuracy'] = metrics_data.get('accuracy', True)
        flat_config['compute_precision'] = metrics_data.get('precision', True)
        flat_config['compute_recall'] = metrics_data.get('recall', True)
        flat_config['compute_f1_score'] = metrics_data.get('f1_score', True)
        flat_config['compute_auc_roc'] = metrics_data.get('auc_roc', True)
        flat_config['compute_auc_pr'] = metrics_data.get('auc_pr', True)
        flat_config['compute_confusion_matrix'] = metrics_data.get('confusion_matrix', True)
        flat_config['compute_sensitivity'] = metrics_data.get('sensitivity', True)
        flat_config['compute_specificity'] = metrics_data.get('specificity', True)
        
        # Plots
        flat_config['plot_confusion_matrix'] = plots_data.get('confusion_matrix', True)
        flat_config['plot_roc_curve'] = plots_data.get('roc_curve', True)
        flat_config['plot_precision_recall_curve'] = plots_data.get('precision_recall_curve', True)
        flat_config['plot_training_history'] = plots_data.get('training_history', True)
        flat_config['plot_attention_visualization'] = plots_data.get('attention_visualization', True)
        
        # Reports
        flat_config['save_per_epoch'] = reports_data.get('save_per_epoch', False)
        flat_config['save_best_model'] = reports_data.get('save_best_model', True)
        flat_config['save_final_test'] = reports_data.get('save_final_test', True)
        
        # Remove nested sections
        for key in ['metrics', 'plots', 'detailed_reports']:
            flat_config.pop(key, None)
        
        return EvaluationConfig(**flat_config)
    
    def _create_hyperparameter_tuning_config(self) -> HyperparameterTuningConfig:
        """Create hyperparameter tuning configuration"""
        tuning_data = self.config_data.get('hyperparameter_tuning', {})
        search_space = tuning_data.get('search_space', {})
        grid_search = tuning_data.get('grid_search', {})
        random_search = tuning_data.get('random_search', {})
        
        # Base config
        flat_config = {**tuning_data}
        
        # Search space
        flat_config['embedding_dims'] = search_space.get('embedding_dim', [64, 128, 256])
        flat_config['hidden_dims'] = search_space.get('hidden_dim', [32, 64, 128, 256])
        flat_config['num_layers_options'] = search_space.get('num_layers', [1, 2, 3])
        flat_config['dropout_values'] = search_space.get('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
        flat_config['learning_rates'] = search_space.get('learning_rate', [0.0001, 0.001, 0.01])
        flat_config['batch_sizes'] = search_space.get('batch_size', [16, 32, 64])
        flat_config['optimizer_types'] = search_space.get('optimizer_type', ["adam", "adamw"])
        flat_config['weight_decays'] = search_space.get('weight_decay', [1e-6, 1e-5, 1e-4])
        flat_config['use_attention_options'] = search_space.get('use_attention', [True, False])
        flat_config['classifier_architectures'] = search_space.get('classifier_layers', [[64, 32], [128, 64, 32], [256, 128, 64]])
        
        # Grid search
        flat_config['grid_search_params'] = grid_search
        
        # Random search
        flat_config['n_random_samples'] = random_search.get('n_random_samples', 50)
        
        # Remove nested sections
        for key in ['search_space', 'grid_search', 'random_search']:
            flat_config.pop(key, None)
        
        return HyperparameterTuningConfig(**flat_config)
    
    def _create_paths_config(self) -> PathsConfig:
        """Create paths configuration"""
        paths_data = self.config_data.get('paths', {})
        return PathsConfig(**paths_data)
    
    def _auto_optimize_for_hardware(self) -> None:
        """Automatically optimize parameters based on available hardware"""
        try:
            # Analyze available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            logger.info(f"🔧 Auto-ottimizzazione hardware: {available_gb:.1f}GB disponibili")
            
            # Adapt parameters based on memory
            if available_gb >= 8:
                # High memory system
                optimized_params = {
                    'batch_size': 32,
                    'embedding_dim': 128,
                    'hidden_dim': 64,
                    'max_samples': 20000,
                    'use_attention': True,
                    'gradient_accumulation_steps': 1
                }
            elif available_gb >= 4:
                # Medium memory system
                optimized_params = {
                    'batch_size': 16,
                    'embedding_dim': 128,
                    'hidden_dim': 64,
                    'max_samples': 15000,
                    'use_attention': True,
                    'gradient_accumulation_steps': 2
                }
            elif available_gb >= 2:
                # Sistema con poca memoria
                optimized_params = {
                    'batch_size': 8,
                    'embedding_dim': 64,
                    'hidden_dim': 32,
                    'max_samples': 10000,
                    'use_attention': False,
                    'gradient_accumulation_steps': 4
                }
            else:
                # Sistema con memoria molto limitata
                optimized_params = {
                    'batch_size': 4,
                    'embedding_dim': 64,
                    'hidden_dim': 32,
                    'max_samples': 5000,
                    'use_attention': False,
                    'gradient_accumulation_steps': 8
                }
            
            # Applica ottimizzazioni
            self.training.batch_size = optimized_params['batch_size']
            self.model.embedding_dim = optimized_params['embedding_dim']
            self.model.hidden_dim = optimized_params['hidden_dim']
            self.data.max_samples = optimized_params['max_samples']
            self.model.use_attention = optimized_params['use_attention']
            self.training.gradient_accumulation_steps = optimized_params['gradient_accumulation_steps']
            
            logger.info(f"✅ Parametri ottimizzati per {available_gb:.1f}GB RAM")
            
        except Exception as e:
            logger.warning(f"Errore auto-ottimizzazione: {e}")
    
    def _setup_reproducibility(self) -> None:
        """Configura la riproducibilità"""
        reproducibility_config = self.config_data.get('reproducibility', {})
        
        seed = reproducibility_config.get('seed', 42)
        deterministic = reproducibility_config.get('deterministic', True)
        benchmark = reproducibility_config.get('benchmark', False)
        
        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configurazioni per riproducibilità
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            torch.backends.cudnn.benchmark = benchmark
        
        logger.info(f"🎲 Riproducibilità configurata: seed={seed}, deterministic={deterministic}")
    
    def _create_directories(self) -> None:
        """Crea le directory necessarie"""
        directories = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.logs_dir,
            self.paths.plots_dir,
            self.paths.results_dir,
            self.paths.tuning_results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("📁 Directory create/verificate")
    
    def update_config_for_tuning(self, trial_params: Dict[str, Any]) -> None:
        """Aggiorna la configurazione con parametri da hyperparameter tuning"""
        for param_name, param_value in trial_params.items():
            # Modello
            if param_name == 'embedding_dim':
                self.model.embedding_dim = param_value
            elif param_name == 'hidden_dim':
                self.model.hidden_dim = param_value
            elif param_name == 'num_layers':
                self.model.num_layers = param_value
            elif param_name == 'dropout':
                self.model.dropout = param_value
            elif param_name == 'use_attention':
                self.model.use_attention = param_value
            elif param_name == 'classifier_layers':
                self.model.classifier_hidden_layers = param_value
            
            # Training
            elif param_name == 'learning_rate':
                self.training.learning_rate = param_value
            elif param_name == 'batch_size':
                self.training.batch_size = param_value
            elif param_name == 'optimizer_type':
                self.training.optimizer_type = param_value
            elif param_name == 'weight_decay':
                self.training.weight_decay = param_value
        
        logger.info(f"🔄 Configurazione aggiornata per trial: {trial_params}")
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Restituisce configurazione modello come dizionario"""
        return {
            'vocab_size': None,  # Sarà impostato dinamicamente
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'dropout': self.model.dropout,
            'use_attention': self.model.use_attention
        }
    
    def get_training_config_dict(self) -> Dict[str, Any]:
        """Restituisce configurazione training come dizionario"""
        return {
            'num_epochs': self.training.num_epochs,
            'batch_size': self.training.batch_size,
            'learning_rate': self.training.learning_rate,
            'weight_decay': self.training.weight_decay,
            'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
            'optimizer_type': self.training.optimizer_type,
            'scheduler_type': self.training.scheduler_type,
            'early_stopping_enabled': self.training.early_stopping_enabled,
            'early_stopping_patience': self.training.early_stopping_patience,
            'early_stopping_metric': self.training.early_stopping_metric
        }
    
    def save_experiment_config(self, experiment_name: str, results: Dict[str, Any]) -> str:
        """Salva configurazione e risultati di un esperimento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.paths.results_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Salva configurazione completa
        config_path = os.path.join(experiment_dir, "config.yaml")
        self.save_config(config_path)
        
        # Salva risultati
        results_path = os.path.join(experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Esperimento salvato in {experiment_dir}")
        return experiment_dir
    
    def __str__(self) -> str:
        """Rappresentazione stringa della configurazione"""
        return f"""
ConfigManager Summary:
- Project: {self.project.name} v{self.project.version}
- Model: {self.model.embedding_dim}d emb, {self.model.hidden_dim}d hidden, {self.model.num_layers} layers
- Training: {self.training.num_epochs} epochs, batch {self.training.batch_size}, lr {self.training.learning_rate}
- Data: {self.data.max_samples} samples, length {self.data.max_length}
- Hardware: auto-opt {self.hardware.auto_memory_management}, dynamic batch {self.hardware.dynamic_batch_size}
- HP Tuning: {self.hyperparameter_tuning.enabled} ({self.hyperparameter_tuning.method})
""" 