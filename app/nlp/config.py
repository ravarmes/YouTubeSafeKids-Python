"""
Configurações centralizadas para o projeto YouTube Safe Kids.

Este módulo contém todas as configurações necessárias para treinamento,
avaliação e uso dos modelos.
"""

import os
from typing import Dict, List, Any

# Configurações de caminhos
PATHS = {
    'project_root': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nlp', 'datasets'),
    'models_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nlp', 'models'),
    'evaluation_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nlp', 'evaluation', 'results'),
    'logs_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nlp', 'logs'),
    'corpus_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nlp', 'datasets', 'corpus.csv')
}

# Configurações das tarefas
TASKS = {
    'AS': {
        'name': 'Análise de Sentimentos',
        'column': 'AS',
        'classes': ['Negativo', 'Neutro', 'Positivo'],
        'num_labels': 3,
        'model_file': 'bertimbau_sentiment.py',
        'filter_file': 'sentiment.py'
    },
    'TOX': {
        'name': 'Toxicidade',
        'column': 'TOX',
        'classes': ['Não Tóxico', 'Levemente Tóxico', 'Moderadamente Tóxico', 'Altamente Tóxico'],
        'num_labels': 4,
        'model_file': 'bertimbau_toxicity.py',
        'filter_file': 'toxicity.py'
    },
    'LI': {
        'name': 'Linguagem Imprópria',
        'column': 'LI',
        'classes': ['Nenhuma', 'Leve', 'Severa'],
        'num_labels': 3,
        'model_file': 'bertimbau_language.py',
        'filter_file': 'language.py'
    },
    'TE': {
        'name': 'Tópicos Educacionais',
        'column': 'TE',
        'classes': ['Não Educacional', 'Parcialmente Educacional', 'Educacional'],
        'num_labels': 3,
        'model_file': 'bertimbau_educational.py',
        'filter_file': 'educational.py'
    }
}

# Configurações do modelo base
MODEL_CONFIG = {
    'base_model': 'neuralmind/bert-base-portuguese-cased',
    'max_length': 128,
    'cache_dir': os.path.join(PATHS['models_dir'], 'cache')
}

# Configurações de treinamento padrão
TRAINING_CONFIG = {
    'default': {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'learning_rate': 5e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'eval_steps': 500,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True,
        'early_stopping_patience': 3,
        'seed': 42
    },
    'fast': {
        'num_train_epochs': 1,
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'learning_rate': 5e-5,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'logging_steps': 5,
        'eval_steps': 100,
        'save_steps': 100,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True,
        'early_stopping_patience': 2,
        'seed': 42
    },
    'extensive': {
        'num_train_epochs': 5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'learning_rate': 2e-5,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'logging_steps': 20,
        'eval_steps': 1000,
        'save_steps': 1000,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True,
        'early_stopping_patience': 5,
        'seed': 42
    }
}

# Configurações de divisão dos dados
DATA_SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'random_state': 42,
    'stratify': True
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
}

# Configurações de avaliação
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'average_methods': ['macro', 'weighted'],
    'save_plots': True,
    'save_predictions': True,
    'confusion_matrix': True,
    'classification_report': True
}

# Configurações de hardware
HARDWARE_CONFIG = {
    'use_cuda': True,
    'mixed_precision': True,
    'dataloader_num_workers': 4,
    'pin_memory': True
}

# Configurações específicas por tarefa (se necessário)
TASK_SPECIFIC_CONFIG = {
    'AS': {
        'class_weights': None,  # Pode ser definido se houver desbalanceamento
        'threshold': 0.5,
        'special_tokens': ['[POSITIVO]', '[NEGATIVO]', '[NEUTRO]']
    },
    'TOX': {
        'class_weights': None,
        'threshold': 0.7,  # Threshold mais alto para toxicidade
        'special_tokens': ['[TOXICO]', '[SEGURO]']
    },
    'LI': {
        'class_weights': None,
        'threshold': 0.6,
        'special_tokens': ['[NENHUMA]', '[SEVERA]']
    },
    'TE': {
        'class_weights': None,
        'threshold': 0.5,
        'special_tokens': ['[EDUCACIONAL]', '[NAO_EDUCACIONAL]']
    }
}


def get_task_config(task: str) -> Dict[str, Any]:
    """
    Retorna configuração específica de uma tarefa.
    
    Args:
        task: Nome da tarefa (AS, TOX, LI, TE)
        
    Returns:
        Dict com configuração da tarefa
    """
    if task not in TASKS:
        raise ValueError(f"Tarefa '{task}' não encontrada. Tarefas disponíveis: {list(TASKS.keys())}")
    
    return TASKS[task]


def get_training_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Retorna configuração de treinamento.
    
    Args:
        config_name: Nome da configuração (default, fast, extensive)
        
    Returns:
        Dict com configuração de treinamento
    """
    if config_name not in TRAINING_CONFIG:
        raise ValueError(f"Configuração '{config_name}' não encontrada. Disponíveis: {list(TRAINING_CONFIG.keys())}")
    
    return TRAINING_CONFIG[config_name].copy()


def get_model_output_dir(task: str, experiment_name: str = None) -> str:
    """
    Gera caminho para salvar modelo.
    
    Args:
        task: Nome da tarefa
        experiment_name: Nome do experimento (opcional)
        
    Returns:
        Caminho do diretório
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{task}_{experiment_name}_{timestamp}"
    else:
        dir_name = f"{task}_{timestamp}"
    
    return os.path.join(PATHS['models_dir'], 'trained', dir_name)


def ensure_directories():
    """Cria diretórios necessários se não existirem."""
    dirs_to_create = [
        PATHS['models_dir'],
        PATHS['evaluation_dir'],
        PATHS['logs_dir'],
        os.path.join(PATHS['models_dir'], 'trained'),
        os.path.join(PATHS['models_dir'], 'cache'),
        os.path.join(PATHS['evaluation_dir'], 'plots'),
        os.path.join(PATHS['evaluation_dir'], 'reports')
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)


def setup_logging(task: str = None):
    """
    Configura logging para o projeto.
    
    Args:
        task: Nome da tarefa (opcional)
    """
    import logging
    from datetime import datetime
    
    # Cria diretório de logs
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    
    # Nome do arquivo de log
    timestamp = datetime.now().strftime("%Y%m%d")
    if task:
        log_file = os.path.join(PATHS['logs_dir'], f"{task}_{timestamp}.log")
    else:
        log_file = os.path.join(PATHS['logs_dir'], f"general_{timestamp}.log")
    
    # Configuração
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Logger específico para arquivos
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG['file_format']))
    
    return logging.getLogger(__name__)


# Inicializa diretórios ao importar o módulo
ensure_directories()