"""
Sistema completo di Hyperparameter Tuning per Tweet Sentiment Analysis
Supporta Optuna, Grid Search, Random Search e configurazioni avanzate
"""

import optuna
import itertools
import random
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from config_manager import ConfigManager
from sentiment_model import SentimentLSTM, SentimentTrainer
from train_sentiment_model import train_model_with_config
import psutil

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Sistema avanzato di hyperparameter tuning"""
    
    def __init__(self, config_manager: ConfigManager, objective_function: Optional[Callable] = None):
        self.config = config_manager
        self.objective_function = objective_function or self._default_objective
        
        # Setup directory per risultati
        self.results_dir = Path(self.config.paths.tuning_results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Timestamp per questa sessione di tuning
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.results_dir / f"session_{self.session_timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Database per tracking risultati
        self.results_db = []
        self.best_result = None
        
        logger.info(f"🔧 HyperparameterTuner inizializzato")
        logger.info(f"📁 Directory sessione: {self.session_dir}")
    
    def run_tuning(self) -> Dict[str, Any]:
        """Esegue hyperparameter tuning secondo la configurazione"""
        if not self.config.hyperparameter_tuning.enabled:
            logger.warning("⚠️ Hyperparameter tuning disabilitato nella configurazione")
            return {}
        
        method = self.config.hyperparameter_tuning.method.lower()
        
        logger.info(f"🚀 Avvio hyperparameter tuning: metodo {method}")
        
        start_time = time.time()
        
        try:
            if method == "optuna":
                results = self._run_optuna_tuning()
            elif method == "grid_search":
                results = self._run_grid_search()
            elif method == "random_search":
                results = self._run_random_search()
            else:
                raise ValueError(f"Metodo non supportato: {method}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Aggiungi statistiche generali
            results['tuning_statistics'] = {
                'method': method,
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'total_trials': len(self.results_db),
                'successful_trials': len([r for r in self.results_db if r.get('success', False)]),
                'best_score': self.best_result['score'] if self.best_result else None,
                'session_timestamp': self.session_timestamp
            }
            
            # Salva risultati completi
            self._save_complete_results(results)
            
            logger.info(f"✅ Hyperparameter tuning completato in {self._format_duration(duration)}")
            if self.best_result:
                logger.info(f"🏆 Miglior risultato: {self.best_result['score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Errore durante hyperparameter tuning: {e}")
            return {'error': str(e)}
    
    def _run_optuna_tuning(self) -> Dict[str, Any]:
        """Esegue tuning con Optuna"""
        logger.info("🔬 Avvio tuning con Optuna")
        
        # Crea studio Optuna
        study_name = f"tweet_sentiment_{self.session_timestamp}"
        study_storage = f"sqlite:///{self.session_dir / 'optuna_study.db'}"
        
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=study_storage,
            load_if_exists=True
        )
        
        # Callback per monitoraggio
        def optuna_callback(study, trial):
            logger.info(f"Trial {trial.number}: score={trial.value:.4f}, params={trial.params}")
        
        # Esegui ottimizzazione
        try:
            study.optimize(
                self._optuna_objective,
                n_trials=self.config.hyperparameter_tuning.n_trials,
                timeout=self.config.hyperparameter_tuning.timeout,
                callbacks=[optuna_callback]
            )
            
            # Risultati
            best_trial = study.best_trial
            
            results = {
                'method': 'optuna',
                'best_params': best_trial.params,
                'best_score': best_trial.value,
                'n_trials': len(study.trials),
                'study_statistics': {
                    'complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                }
            }
            
            # Salva visualizzazioni Optuna
            self._save_optuna_visualizations(study)
            
            return results
            
        except Exception as e:
            logger.error(f"Errore Optuna tuning: {e}")
            return {'error': str(e)}
    
    def _optuna_objective(self, trial) -> float:
        """Funzione obiettivo per Optuna"""
        # Suggerisci parametri da ottimizzare
        params = {}
        
        # Architettura
        params['embedding_dim'] = trial.suggest_categorical(
            'embedding_dim', 
            self.config.hyperparameter_tuning.embedding_dims
        )
        params['hidden_dim'] = trial.suggest_categorical(
            'hidden_dim',
            self.config.hyperparameter_tuning.hidden_dims
        )
        params['num_layers'] = trial.suggest_categorical(
            'num_layers',
            self.config.hyperparameter_tuning.num_layers_options
        )
        params['dropout'] = trial.suggest_categorical(
            'dropout',
            self.config.hyperparameter_tuning.dropout_values
        )
        params['use_attention'] = trial.suggest_categorical(
            'use_attention',
            self.config.hyperparameter_tuning.use_attention_options
        )
        
        # Training
        params['learning_rate'] = trial.suggest_categorical(
            'learning_rate',
            self.config.hyperparameter_tuning.learning_rates
        )
        params['optimizer_type'] = trial.suggest_categorical(
            'optimizer_type',
            self.config.hyperparameter_tuning.optimizer_types
        )
        params['weight_decay'] = trial.suggest_categorical(
            'weight_decay',
            self.config.hyperparameter_tuning.weight_decays
        )
        
        # Batch size (se memoria sufficiente)
        memory_gb = psutil.virtual_memory().available / (1024**3)
        available_batch_sizes = [bs for bs in self.config.hyperparameter_tuning.batch_sizes if bs <= self._max_batch_size_for_memory(memory_gb)]
        
        if available_batch_sizes:
            params['batch_size'] = trial.suggest_categorical('batch_size', available_batch_sizes)
        
        # Classificatore architettura
        params['classifier_layers'] = trial.suggest_categorical(
            'classifier_layers',
            self.config.hyperparameter_tuning.classifier_architectures
        )
        
        return self.objective_function(params, trial.number)
    
    def _run_grid_search(self) -> Dict[str, Any]:
        """Esegue Grid Search"""
        logger.info("🔬 Avvio Grid Search")
        
        # Crea griglia parametri
        param_grid = self.config.hyperparameter_tuning.grid_search_params
        
        # Genera tutte le combinazioni
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"📊 Grid Search: {len(combinations)} combinazioni da testare")
        
        best_score = -float('inf')
        best_params = None
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                score = self.objective_function(params, i)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Grid {i+1}/{len(combinations)}: score={score:.4f}, params={params}")
                
            except Exception as e:
                logger.error(f"Errore grid combination {i}: {e}")
        
        return {
            'method': 'grid_search',
            'best_params': best_params,
            'best_score': best_score,
            'total_combinations': len(combinations)
        }
    
    def _run_random_search(self) -> Dict[str, Any]:
        """Esegue Random Search"""
        logger.info("🔬 Avvio Random Search")
        
        n_samples = self.config.hyperparameter_tuning.n_random_samples
        logger.info(f"🎲 Random Search: {n_samples} campioni casuali")
        
        best_score = -float('inf')
        best_params = None
        
        for i in range(n_samples):
            params = self._sample_random_params()
            
            try:
                score = self.objective_function(params, i)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Random {i+1}/{n_samples}: score={score:.4f}, params={params}")
                
            except Exception as e:
                logger.error(f"Errore random sample {i}: {e}")
        
        return {
            'method': 'random_search',
            'best_params': best_params,
            'best_score': best_score,
            'total_samples': n_samples
        }
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Campiona parametri casuali"""
        params = {}
        
        params['embedding_dim'] = random.choice(self.config.hyperparameter_tuning.embedding_dims)
        params['hidden_dim'] = random.choice(self.config.hyperparameter_tuning.hidden_dims)
        params['num_layers'] = random.choice(self.config.hyperparameter_tuning.num_layers_options)
        params['dropout'] = random.choice(self.config.hyperparameter_tuning.dropout_values)
        params['learning_rate'] = random.choice(self.config.hyperparameter_tuning.learning_rates)
        params['optimizer_type'] = random.choice(self.config.hyperparameter_tuning.optimizer_types)
        params['weight_decay'] = random.choice(self.config.hyperparameter_tuning.weight_decays)
        params['use_attention'] = random.choice(self.config.hyperparameter_tuning.use_attention_options)
        params['classifier_layers'] = random.choice(self.config.hyperparameter_tuning.classifier_architectures)
        
        # Batch size (controlla memoria)
        memory_gb = psutil.virtual_memory().available / (1024**3)
        available_batch_sizes = [bs for bs in self.config.hyperparameter_tuning.batch_sizes if bs <= self._max_batch_size_for_memory(memory_gb)]
        
        if available_batch_sizes:
            params['batch_size'] = random.choice(available_batch_sizes)
        
        return params
    
    def _default_objective(self, params: Dict[str, Any], trial_number: int) -> float:
        """Funzione obiettivo di default"""
        logger.info(f"🔄 Trial {trial_number}: testing {params}")
        
        try:
            # Crea una copia della configurazione
            trial_config = ConfigManager(self.config.config_path)
            
            # Aggiorna con parametri del trial
            trial_config.update_config_for_tuning(params)
            
            # Addestra modello
            results = train_model_with_config(trial_config)
            
            # Estrai metrica principale
            score = results.get('best_val_f1', results.get('best_val_acc', 0))
            
            # Salva risultati trial
            trial_result = {
                'trial_number': trial_number,
                'params': params,
                'score': score,
                'results': results,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results_db.append(trial_result)
            
            # Aggiorna miglior risultato
            if self.best_result is None or score > self.best_result['score']:
                self.best_result = trial_result.copy()
                
                # Salva miglior configurazione
                best_config_path = self.session_dir / f"best_config_trial_{trial_number}.yaml"
                trial_config.save_config(str(best_config_path))
                
                logger.info(f"🏆 Nuovo miglior risultato: {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Errore trial {trial_number}: {e}")
            
            # Salva errore
            error_result = {
                'trial_number': trial_number,
                'params': params,
                'score': 0.0,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results_db.append(error_result)
            
            return 0.0
    
    def _max_batch_size_for_memory(self, memory_gb: float) -> int:
        """Calcola batch size massimo per memoria disponibile"""
        if memory_gb >= 8:
            return 64
        elif memory_gb >= 4:
            return 32
        elif memory_gb >= 2:
            return 16
        else:
            return 8
    
    def _save_optuna_visualizations(self, study) -> None:
        """Salva visualizzazioni Optuna"""
        try:
            import plotly
            
            # Storia dell'ottimizzazione
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(str(self.session_dir / "optimization_history.html"))
            
            # Importanza parametri
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(self.session_dir / "param_importances.html"))
            
            # Slice plot
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(str(self.session_dir / "param_slice.html"))
            
            # Parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(str(self.session_dir / "parallel_coordinate.html"))
            
            logger.info("📊 Visualizzazioni Optuna salvate")
            
        except ImportError:
            logger.warning("plotly non installato, visualizzazioni Optuna non disponibili")
        except Exception as e:
            logger.error(f"Errore salvataggio visualizzazioni: {e}")
    
    def _save_complete_results(self, results: Dict[str, Any]) -> None:
        """Salva risultati completi della sessione"""
        # Risultati principali
        results_file = self.session_dir / "tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Database completo dei trial
        db_file = self.session_dir / "trials_database.json"
        with open(db_file, 'w') as f:
            json.dump(self.results_db, f, indent=2, default=str)
        
        # Statistiche e report
        self._generate_analysis_report()
        
        logger.info(f"💾 Risultati completi salvati in {self.session_dir}")
    
    def _generate_analysis_report(self) -> None:
        """Genera report di analisi dettagliato"""
        if not self.results_db:
            return
        
        # Converti a DataFrame per analisi
        df = pd.DataFrame(self.results_db)
        
        # Report testuale
        report_lines = [
            "# HYPERPARAMETER TUNING - REPORT ANALISI",
            f"Sessione: {self.session_timestamp}",
            f"Metodo: {self.config.hyperparameter_tuning.method}",
            "",
            "## STATISTICHE GENERALI",
            f"- Total trials: {len(df)}",
            f"- Successful trials: {len(df[df['success'] == True])}",
            f"- Failed trials: {len(df[df['success'] == False])}",
        ]
        
        if 'score' in df.columns and len(df[df['success'] == True]) > 0:
            successful_df = df[df['success'] == True]
            
            report_lines.extend([
                "",
                "## STATISTICHE PERFORMANCE",
                f"- Best score: {successful_df['score'].max():.4f}",
                f"- Mean score: {successful_df['score'].mean():.4f}",
                f"- Std score: {successful_df['score'].std():.4f}",
                f"- Min score: {successful_df['score'].min():.4f}",
            ])
            
            # Top 5 risultati
            top_5 = successful_df.nlargest(5, 'score')
            report_lines.extend([
                "",
                "## TOP 5 RISULTATI",
            ])
            
            for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                report_lines.append(f"{i}. Trial {row['trial_number']}: {row['score']:.4f}")
                params_str = ", ".join([f"{k}={v}" for k, v in row['params'].items()])
                report_lines.append(f"   Params: {params_str}")
            
            # Analisi parametri (se abbastanza dati)
            if len(successful_df) >= 10:
                report_lines.extend([
                    "",
                    "## ANALISI PARAMETRI",
                ])
                
                # Estrai parametri in colonne separate
                params_df = pd.json_normalize(successful_df['params'])
                
                for param in params_df.columns:
                    if params_df[param].dtype in ['object', 'bool']:
                        # Parametri categorici
                        value_counts = params_df[param].value_counts()
                        best_value = params_df.loc[successful_df['score'].idxmax(), param]
                        report_lines.append(f"- {param}: migliore={best_value}, distribuzione={dict(value_counts)}")
                    else:
                        # Parametri numerici
                        correlation = np.corrcoef(params_df[param], successful_df['score'])[0, 1]
                        best_value = params_df.loc[successful_df['score'].idxmax(), param]
                        report_lines.append(f"- {param}: migliore={best_value}, correlazione={correlation:.3f}")
        
        # Salva report
        report_file = self.session_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Salva anche CSV per analisi ulteriori
        if 'params' in df.columns:
            params_df = pd.json_normalize(df['params'])
            analysis_df = pd.concat([
                df[['trial_number', 'score', 'success', 'timestamp']],
                params_df
            ], axis=1)
            
            csv_file = self.session_dir / "trials_analysis.csv"
            analysis_df.to_csv(csv_file, index=False)
        
        logger.info("📋 Report di analisi generato")
    
    def _format_duration(self, seconds: float) -> str:
        """Formatta durata in formato leggibile"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def get_best_config(self) -> Optional[ConfigManager]:
        """Restituisce la migliore configurazione trovata"""
        if not self.best_result:
            return None
        
        # Crea configurazione con migliori parametri
        best_config = ConfigManager(self.config.config_path)
        best_config.update_config_for_tuning(self.best_result['params'])
        
        return best_config
    
    def load_previous_session(self, session_timestamp: str) -> Dict[str, Any]:
        """Carica risultati da sessione precedente"""
        session_dir = self.results_dir / f"session_{session_timestamp}"
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Sessione {session_timestamp} non trovata")
        
        results_file = session_dir / "tuning_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        db_file = session_dir / "trials_database.json"
        with open(db_file, 'r') as f:
            self.results_db = json.load(f)
        
        return results

# La funzione train_model_with_config è ora importata da train_sentiment_model.py

def main():
    """Funzione principale per testing del sistema"""
    print("🧪 SISTEMA HYPERPARAMETER TUNING - DEMO")
    print("="*60)
    
    # Carica configurazione
    config = ConfigManager("config.yaml")
    
    # Abilita hyperparameter tuning per demo
    config.hyperparameter_tuning.enabled = True
    config.hyperparameter_tuning.method = "random_search"
    config.hyperparameter_tuning.n_random_samples = 5  # Demo veloce
    
    print(f"📋 Configurazione caricata")
    print(f"   Metodo: {config.hyperparameter_tuning.method}")
    print(f"   Enabled: {config.hyperparameter_tuning.enabled}")
    
    # Crea tuner
    tuner = HyperparameterTuner(config)
    
    # Esegui tuning
    results = tuner.run_tuning()
    
    print("\n📊 RISULTATI:")
    print(f"   Metodo: {results.get('method', 'N/A')}")
    print(f"   Miglior score: {results.get('best_score', 'N/A')}")
    print(f"   Migliori parametri: {results.get('best_params', 'N/A')}")
    
    if 'tuning_statistics' in results:
        stats = results['tuning_statistics']
        print(f"   Durata: {stats['duration_formatted']}")
        print(f"   Trial totali: {stats['total_trials']}")
        print(f"   Trial riusciti: {stats['successful_trials']}")
    
    print("\n✅ Demo completata!")

if __name__ == "__main__":
    main() 