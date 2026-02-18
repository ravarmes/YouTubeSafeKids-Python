"""
Utilitários para processamento de dados do corpus.

Este módulo fornece funções para carregar, processar e dividir o dataset
de acordo com as diferentes tarefas de classificação.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Any, Generator
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Classe para processamento de dados do corpus."""
    
    def __init__(self, corpus_path: str):
        """
        Inicializa o processador de dados.
        
        Args:
            corpus_path: Caminho para o arquivo corpus.csv
        """
        self.corpus_path = corpus_path
        self.df = None
        
    def load_corpus(self) -> pd.DataFrame:
        """
        Carrega o corpus do arquivo CSV.
        
        Returns:
            DataFrame com os dados do corpus
        """
        logger.info(f"Carregando corpus de {self.corpus_path}")
        self.df = pd.read_csv(self.corpus_path, sep=';')
        logger.info(f"Corpus carregado com {len(self.df)} registros")
        return self.df
    
    def get_task_data(self, task: str) -> Tuple[List[str], List[int]]:
        """
        Extrai dados para uma tarefa específica.
        
        Args:
            task: Nome da tarefa ('AS', 'TOX', 'LI', 'TE')
            
        Returns:
            Tuple com textos e labels para a tarefa
        """
        if self.df is None:
            self.load_corpus()
            
        # Remove registros com valores nulos na coluna da tarefa
        task_df = self.df.dropna(subset=[task])
        
        texts = task_df['FRASE'].tolist()
        labels = task_df[task].tolist()
        
        logger.info(f"Dados extraídos para tarefa {task}: {len(texts)} amostras")
        logger.info(f"Distribuição de classes: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def split_data(
        self, 
        texts: List[str], 
        labels: List[int], 
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        Divide os dados em conjuntos de treino, validação e teste.
        
        Args:
            texts: Lista de textos
            labels: Lista de labels
            train_size: Proporção para treino (padrão: 0.8)
            val_size: Proporção para validação (padrão: 0.1)
            test_size: Proporção para teste (padrão: 0.1)
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Verifica se as proporções somam 1
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("As proporções devem somar 1.0")
        
        # Primeira divisão: treino vs (validação + teste)
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, 
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=labels
        )
        
        # Segunda divisão: validação vs teste
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Divisão dos dados:")
        logger.info(f"  Treino: {len(X_train)} amostras")
        logger.info(f"  Validação: {len(X_val)} amostras")
        logger.info(f"  Teste: {len(X_test)} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset_by_task(corpus_path: str, task: str) -> Dict[str, Any]:
    """
    Função de conveniência para dividir o dataset por tarefa.
    
    Args:
        corpus_path: Caminho para o arquivo corpus.csv
        task: Nome da tarefa ('AS', 'TOX', 'LI', 'TE')
        
    Returns:
        Dict com os dados divididos
    """
    processor = DataProcessor(corpus_path)
    texts, labels = processor.get_task_data(task)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(texts, labels)
    
    return {
        'train': {'texts': X_train, 'labels': y_train},
        'validation': {'texts': X_val, 'labels': y_val},
        'test': {'texts': X_test, 'labels': y_test},
        'num_labels': len(set(labels)),
        'label_distribution': pd.Series(labels).value_counts().to_dict()
    }


def get_task_info() -> Dict[str, Dict[str, Any]]:
    """
    Retorna informações sobre as tarefas disponíveis.
    
    Returns:
        Dict com informações das tarefas
    """
    return {
        'AS': {
            'name': 'Análise de Sentimentos',
            'description': 'Classifica textos em sentimentos: negativo (0), neutro (1), positivo (2)',
            'num_labels': 3,
            'labels': {0: 'negativo', 1: 'neutro', 2: 'positivo'}
        },
        'TOX': {
            'name': 'Detecção de Toxicidade',
            'description': 'Classifica textos por nível de toxicidade: nenhuma (0), leve (1), severa (2)',
            'num_labels': 3,
            'labels': {0: 'nenhuma', 1: 'leve', 2: 'severa'}
        },
        'LI': {
            'name': 'Linguagem Imprópria',
            'description': 'Classifica adequação da linguagem: nenhuma (0), leve (1), severa (2)',
        'type': 'classification',
        'labels': {0: 'nenhuma', 1: 'leve', 2: 'severa'}
        },
        'TE': {
            'name': 'Tópicos Educacionais',
            'description': 'Classifica valor educacional: não-educacional (0), baixo (1), médio (2), alto (3)',
            'num_labels': 4,
            'labels': {0: 'não-educacional', 1: 'baixo', 2: 'médio', 3: 'alto'}
        }
    }

# --- NOVA FUNÇÃO PARA CROSS-VALIDATION ---
def get_kfold_split(texts: np.ndarray, labels: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Generator:
    """
    Gera índices para Cross-Validation estratificado.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(texts, labels)