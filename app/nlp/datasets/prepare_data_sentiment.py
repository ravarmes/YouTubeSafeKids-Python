"""
Script de preparação de dados para Análise de Sentimentos (AS).
Configurado conforme metodologia:
- 80% Treino (Usado para Cross-Validation)
- 20% Teste (Reservado para avaliação final)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.config import PATHS
from app.nlp.utils.data_utils import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAPPING = {'Negativo': 0, 'Neutro': 1, 'Positivo': 2}

def map_labels_to_numbers(labels: list) -> list:
    numeric_labels = []
    for label in labels:
        if isinstance(label, str):
            numeric_labels.append(LABEL_MAPPING.get(label, -1))
        else:
            numeric_labels.append(label)
    return numeric_labels

def get_data_for_cv_and_test(test_size: float = 0.20):
    """
    Retorna:
      - (X_train_cv, y_train_cv): 80% dos dados para rodar o Cross-Validation.
      - (X_test, y_test): 20% dos dados reservados para teste final.
    """
    corpus_path = PATHS['corpus_file']
    processor = DataProcessor(corpus_path)
    processor.load_corpus()
    
    texts, labels = processor.get_task_data('AS')
    numeric_labels = map_labels_to_numbers(labels)
    
    # Limpeza de labels inválidos
    valid_indices = [i for i, label in enumerate(numeric_labels) if label != -1]
    texts = [texts[i] for i in valid_indices]
    numeric_labels = [numeric_labels[i] for i in valid_indices]
    
    # Divisão: 80% Treino (para CV) vs 20% Teste (Final)
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(
        texts, numeric_labels,
        test_size=test_size,
        random_state=42,
        stratify=numeric_labels
    )
    
    logger.info(f"=== METODOLOGIA DE DIVISÃO ===")
    logger.info(f"Total de Amostras: {len(texts)}")
    logger.info(f"Conjunto de TREINO (Para Cross-Validation): {len(X_train_cv)} amostras ({(1-test_size)*100:.0f}%)")
    logger.info(f"Conjunto de TESTE (Reservado): {len(X_test)} amostras ({test_size*100:.0f}%)")
    
    return (X_train_cv, y_train_cv), (X_test, y_test)

if __name__ == "__main__":
    # Teste rápido para ver se os números batem
    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    print(f"\nCheck Distribuição Teste: {pd.Series(y_test).value_counts().to_dict()}")