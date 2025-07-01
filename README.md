# Tweet Sentiment Analysis with Neural Networks

This project implements an LSTM neural network for sentiment analysis of tweets, classifying them as **positive** or **negative**. It features a complete system for configuration, training, hyperparameter tuning, and deployment.

## Key Features

- **Bidirectional LSTM Network**: Captures long-term dependencies in text for accurate sentiment analysis.
- **External YAML Configuration**: All parameters are managed via `config.yaml` and `config_dev.yaml`, eliminating hardcoded values.
- **Hyperparameter Tuning**: Integrated support for `Optuna`, `Grid Search`, and `Random Search` to find the best model parameters automatically.
- **Hardware Auto-Optimization**: The system automatically adjusts batch size and other parameters based on available system memory (RAM).
- **Professional CLI**: A user-friendly command-line interface built with `Rich` for running training, tuning, and validation.
- **Automated Data Preprocessing**: Cleans and prepares tweet data, including URL removal, tokenization, and vocabulary creation.
- **GPU Acceleration Support**: Automatically utilizes available GPUs for faster training.

## Requirements

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
Main dependencies include:
- PyTorch (>= 2.0.0)
- Optuna
- Rich
- PyYAML
- Pandas, NumPy, Scikit-learn

## Project Structure

```
tweet-ML/
├── config.yaml                     # Main production configuration
├── config_dev.yaml                 # Development configuration for quick tests
├── config_manager.py               # Manages loading and validation of configs
├── run_hyperparameter_tuning.py    # Main entry point to run the system
├── hyperparameter_tuning.py        # Implements tuning methods (Optuna, etc.)
├── train_sentiment_model.py        # Core training loop logic
├── sentiment_model.py              # Neural network architecture
├── data_preparation.py             # Data loading and preprocessing scripts
├── test_model.py                   # Script for evaluating the trained model
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── data/                           # Processed datasets (auto-generated)
└── models/                         # Trained models and vocabularies (auto-generated)
```

## How to Use

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running the System

The main entry point is `run_hyperparameter_tuning.py`, which supports several modes.

#### Validate Configuration
Check if your `config.yaml` is valid without running a full training session:
```bash
python run_hyperparameter_tuning.py --config config.yaml --validate
```

#### Run a Single Training Session
Train the model using the parameters defined in a configuration file (`config_dev.yaml` for a quick test):
```bash
python run_hyperparameter_tuning.py --config config_dev.yaml --mode training
```
For a full training session using the production configuration:
```bash
python run_hyperparameter_tuning.py --config config.yaml --mode training
```

#### Run Hyperparameter Tuning
Start a hyperparameter tuning session using the method and search space defined in `config.yaml`:
```bash
python run_hyperparameter_tuning.py --config config.yaml --mode tuning --method optuna
```
You can specify the number of trials:
```bash
python run_hyperparameter_tuning.py --config config.yaml --mode tuning --method optuna --trials 50
```

### 3. Testing the Model
After a model is trained, you can test it using `test_model.py`:
```bash
python test_model.py
```
This script provides three modes:
1.  **Test on predefined examples**: Runs inference on a sample set of tweets.
2.  **Interactive mode**: Enter your own tweets to get sentiment predictions.
3.  **CSV file analysis**: Provide a path to a CSV file to analyze a batch of tweets.


## Neural Network Architecture

```
Input Tweet -> Preprocessing -> Tokenization -> Embedding Layer
                                                    |
                                           Bidirectional LSTM (2 layers)
                                                    |
                                                 Attention Mechanism
                                                    |
                                               Pooling Layers
                                                    |
                                           Fully Connected Classifier
                                                    |
                                           Output: [Negative, Positive]
```

-   **Embedding Layer**: Converts input tokens into dense vectors.
-   **Bidirectional LSTM**: Processes text in both forward and backward directions to capture context.
-   **Attention Mechanism**: Allows the model to focus on the most relevant words for sentiment.
-   **Classifier**: A series of fully connected layers with dropout for final classification.

## Customization

All aspects of the project are controlled via the `config.yaml` and `config_dev.yaml` files.

### Modifying Model and Training Parameters
Edit the `config.yaml` file to change any parameter, such as:
-   `model.embedding_dim`, `model.hidden_dim`
-   `training.num_epochs`, `training.learning_rate`, `training.batch_size`
-   `hyperparameter_tuning.search_space`

### Using Custom Datasets
Modify the `data_preparation.py` script to load your own dataset. Ensure it is formatted with `text` and `sentiment` columns.

## Troubleshooting

### Insufficient Memory
The system is designed to auto-optimize for available RAM. If you encounter memory issues, you can manually adjust these parameters in `config.yaml`:
-   `training.batch_size` (e.g., from 32 to 16)
-   `data.max_samples` (e.g., from 20000 to 10000)
-   `model.hidden_dim` (e.g., from 128 to 64)

### GPU Not Detected
The model will fall back to CPU if a GPU is not available. To ensure PyTorch is installed with GPU support, follow the official instructions on the PyTorch website for your specific CUDA version.

## Datasets

The project uses publicly available datasets, such as the **Emotion Dataset** from HuggingFace, which contains tweets labeled with various emotions. It also includes a synthetic dataset generator as a fallback for testing purposes.

## Contributing

1.  Fork the repository.
2.  Create a new branch for your feature.
3.  Commit your changes.
4.  Push to your branch and open a Pull Request.

## License

This project is released under the MIT License. 