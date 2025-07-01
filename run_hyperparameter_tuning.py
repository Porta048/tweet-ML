#!/usr/bin/env python3
"""
TWEET SENTIMENT ANALYSIS - HYPERPARAMETER TUNING
Launcher principale per l'ottimizzazione degli iperparametri
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Setup logging colorato
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    import rich
    
    console = Console()
    
    # Configurazione logging con Rich
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
except ImportError:
    # Fallback a logging standard
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Import dei moduli principali
try:
    from config_manager import ConfigManager
    from hyperparameter_tuning import HyperparameterTuner
    from train_sentiment_model import train_model_with_config
    from train_sentiment_model import main as train_main
except ImportError as e:
    logger.error(f"❌ Errore import: {e}")
    logger.error("Assicurati che tutti i file necessari siano presenti")
    sys.exit(1)

def create_rich_header():
    """Crea header Rich formattato"""
    try:
        header = Panel(
            "[bold blue]🚀 TWEET SENTIMENT ANALYSIS[/bold blue]\n"
            "[cyan]Sistema Avanzato di Hyperparameter Tuning[/cyan]\n"
            "[dim]Supporta Optuna, Grid Search, Random Search[/dim]",
            title="[bold green]ML Pipeline[/bold green]",
            border_style="blue"
        )
        console.print(header)
    except:
        print("="*60)
        print("🚀 TWEET SENTIMENT ANALYSIS - HYPERPARAMETER TUNING")
        print("="*60)

def display_config_summary(config: ConfigManager):
    """Mostra riassunto configurazione"""
    try:
        table = Table(title="📋 Configurazione Attuale")
        table.add_column("Parametro", style="cyan")
        table.add_column("Valore", style="green")
        
        # Informazioni principali
        table.add_row("Progetto", f"{config.project.name} v{config.project.version}")
        table.add_row("Dataset", f"{config.data.max_samples:,} campioni, lunghezza {config.data.max_length}")
        table.add_row("Modello", f"Emb:{config.model.embedding_dim}, Hidden:{config.model.hidden_dim}, Layers:{config.model.num_layers}")
        table.add_row("Training", f"Epochs:{config.training.num_epochs}, Batch:{config.training.batch_size}, LR:{config.training.learning_rate}")
        table.add_row("Attention", "✅" if config.model.use_attention else "❌")
        table.add_row("HP Tuning", "✅" if config.hyperparameter_tuning.enabled else "❌")
        
        if config.hyperparameter_tuning.enabled:
            table.add_row("Metodo", config.hyperparameter_tuning.method.upper())
            if config.hyperparameter_tuning.method == "optuna":
                table.add_row("N Trials", str(config.hyperparameter_tuning.n_trials))
            elif config.hyperparameter_tuning.method == "random_search":
                table.add_row("N Samples", str(config.hyperparameter_tuning.n_random_samples))
        
        console.print(table)
    except:
        # Fallback testuale
        print("\n📋 CONFIGURAZIONE ATTUALE:")
        print(f"   Progetto: {config.project.name} v{config.project.version}")
        print(f"   Dataset: {config.data.max_samples:,} campioni")
        print(f"   Modello: {config.model.embedding_dim}d emb, {config.model.hidden_dim}d hidden")
        print(f"   Training: {config.training.num_epochs} epochs, batch {config.training.batch_size}")
        print(f"   HP Tuning: {'✅' if config.hyperparameter_tuning.enabled else '❌'}")

def run_single_training(config_path: str, save_results: bool = True):
    """Esegue training singolo con configurazione"""
    logger.info("🎯 Avvio training singolo")
    
    try:
        config = ConfigManager(config_path)
        
        # Verifica che hyperparameter tuning sia disabilitato
        if config.hyperparameter_tuning.enabled:
            logger.warning("⚠️ Hyperparameter tuning abilitato, lo disabilito per training singolo")
            config.hyperparameter_tuning.enabled = False
        
                # Addestra modello
        results = train_model_with_config(config)
        
        if save_results:
            # Salva risultati
            experiment_dir = config.save_experiment_config("single_training", results)
            logger.info(f"💾 Risultati salvati in {experiment_dir}")
        
        # Mostra risultati
        display_training_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Errore training singolo: {e}")
        return None

def run_hyperparameter_tuning(config_path: str, method: str = None, n_trials: int = None):
    """Esegue hyperparameter tuning"""
    logger.info("🔬 Avvio hyperparameter tuning")
    
    try:
        config = ConfigManager(config_path)
        
        # Override parametri se specificati
        if method:
            config.hyperparameter_tuning.method = method
        if n_trials:
            if method == "optuna":
                config.hyperparameter_tuning.n_trials = n_trials
            elif method == "random_search":
                config.hyperparameter_tuning.n_random_samples = n_trials
        
        # Abilita hyperparameter tuning
        config.hyperparameter_tuning.enabled = True
        
        # Crea tuner
        tuner = HyperparameterTuner(config)
        
        # Esegui tuning
        with console.status("[bold green]Esecuzione hyperparameter tuning...") if 'console' in globals() else Progress() as status:
            results = tuner.run_tuning()
        
        # Mostra risultati
        display_tuning_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Errore hyperparameter tuning: {e}")
        return None

def display_training_results(results: dict):
    """Mostra risultati training"""
    if not results or 'error' in results:
        logger.error("❌ Training fallito")
        return
    
    try:
        table = Table(title="📊 Risultati Training")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green")
        
        # Aggiungi metriche disponibili
        metrics_map = {
            'best_val_f1': 'F1-Score',
            'best_val_acc': 'Accuracy',
            'best_val_precision': 'Precision',
            'best_val_recall': 'Recall',
            'final_epoch': 'Epoche',
            'training_time': 'Tempo (s)'
        }
        
        for key, label in metrics_map.items():
            if key in results:
                value = results[key]
                if isinstance(value, float):
                    if key == 'training_time':
                        table.add_row(label, f"{value:.1f}")
                    else:
                        table.add_row(label, f"{value:.4f}")
                else:
                    table.add_row(label, str(value))
        
        console.print(table)
        
    except:
        # Fallback testuale
        print("\n📊 RISULTATI TRAINING:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

def display_tuning_results(results: dict):
    """Mostra risultati hyperparameter tuning"""
    if not results or 'error' in results:
        logger.error("❌ Hyperparameter tuning fallito")
        return
    
    try:
        # Tabella risultati principali
        table = Table(title="🏆 Risultati Hyperparameter Tuning")
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green")
        
        table.add_row("Metodo", results.get('method', 'N/A').upper())
        
        if 'best_score' in results:
            table.add_row("Miglior Score", f"{results['best_score']:.4f}")
        
        if 'tuning_statistics' in results:
            stats = results['tuning_statistics']
            table.add_row("Durata", stats.get('duration_formatted', 'N/A'))
            table.add_row("Total Trials", str(stats.get('total_trials', 'N/A')))
            table.add_row("Successful Trials", str(stats.get('successful_trials', 'N/A')))
        
        console.print(table)
        
        # Tabella migliori parametri
        if 'best_params' in results and results['best_params']:
            params_table = Table(title="🔧 Migliori Parametri")
            params_table.add_column("Parametro", style="yellow")
            params_table.add_column("Valore", style="green")
            
            for param, value in results['best_params'].items():
                params_table.add_row(param, str(value))
            
            console.print(params_table)
        
    except:
        # Fallback testuale
        print("\n🏆 RISULTATI HYPERPARAMETER TUNING:")
        print(f"   Metodo: {results.get('method', 'N/A')}")
        print(f"   Miglior score: {results.get('best_score', 'N/A')}")
        
        if 'best_params' in results:
            print("\n   Migliori parametri:")
            for param, value in results['best_params'].items():
                print(f"     {param}: {value}")

def validate_config(config_path: str) -> bool:
    """Valida file di configurazione"""
    if not os.path.exists(config_path):
        logger.error(f"❌ File configurazione non trovato: {config_path}")
        return False
    
    try:
        config = ConfigManager(config_path)
        logger.info("✅ Configurazione valida")
        return True
    except Exception as e:
        logger.error(f"❌ Errore configurazione: {e}")
        return False

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(
        description="Sistema di Hyperparameter Tuning per Tweet Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  %(prog)s --config config.yaml --mode training
  %(prog)s --config config.yaml --mode tuning --method optuna --trials 50
  %(prog)s --config config.yaml --mode tuning --method random_search --trials 20
  %(prog)s --config config.yaml --validate
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Percorso file di configurazione (default: config.yaml)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["training", "tuning", "validate"],
        default="training",
        help="Modalità di esecuzione (default: training)"
    )
    
    parser.add_argument(
        "--method",
        choices=["optuna", "grid_search", "random_search"],
        help="Metodo hyperparameter tuning (override config)"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=int,
        help="Numero di trial/samples (override config)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Output ridotto"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Valida solo la configurazione"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Header
    if not args.quiet:
        create_rich_header()
    
    # Valida configurazione
    if args.validate or args.mode == "validate":
        if validate_config(args.config):
            config = ConfigManager(args.config)
            if not args.quiet:
                display_config_summary(config)
            logger.info("✅ Configurazione valida!")
            return 0
        else:
            return 1
    
    # Controlla che il file config esista
    if not validate_config(args.config):
        return 1
    
    # Carica e mostra configurazione
    config = ConfigManager(args.config)
    if not args.quiet:
        display_config_summary(config)
    
    # Esegui modalità richiesta
    try:
        if args.mode == "training":
            logger.info("🎯 Modalità: Training singolo")
            results = run_single_training(args.config)
            
        elif args.mode == "tuning":
            logger.info("🔬 Modalità: Hyperparameter tuning")
            results = run_hyperparameter_tuning(
                args.config, 
                method=args.method, 
                n_trials=args.trials
            )
        
        if results:
            logger.info("✅ Operazione completata con successo!")
            return 0
        else:
            logger.error("❌ Operazione fallita")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Operazione interrotta dall'utente")
        return 1
    except Exception as e:
        logger.error(f"❌ Errore critico: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 