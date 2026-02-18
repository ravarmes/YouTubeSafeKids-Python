"""
Script de preparação de dados para Detecção de Toxicidade (TOX).
Configurado conforme metodologia:
- 80% Treino (Usado para Cross-Validation)
- 20% Teste (Reservado para avaliação final)

Formato do corpus: data/corpus_toxicidade.csv
Colunas: ID, Nome, Frase, Parte, Link, Categoria Principal, Tox
Classes: 0=Nenhuma, 1=Leve, 2=Severa
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.config import PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapeamento de labels (já estão numéricos no corpus)
LABEL_MAPPING = {0: 0, 1: 1, 2: 2}


def load_toxicity_corpus() -> pd.DataFrame:
    """
    Carrega o corpus de toxicidade.
    
    Returns:
        DataFrame com as colunas relevantes
    """
    corpus_path = PATHS['corpus_toxicidade']
    logger.info(f"Carregando corpus de toxicidade de: {corpus_path}")
    
    df = pd.read_csv(corpus_path)
    logger.info(f"Corpus carregado: {len(df)} registros")
    
    # Remove registros com Tox nulo
    df = df.dropna(subset=['Tox'])
    
    # Converte Tox para int
    df['Tox'] = df['Tox'].astype(int)
    
    logger.info(f"Após limpeza: {len(df)} registros")
    logger.info(f"Distribuição de classes:\n{df['Tox'].value_counts().sort_index()}")
    
    return df


def get_data_for_cv_and_test(test_size: float = 0.20):
    """
    Prepara dados para Cross-Validation e teste final.
    
    Args:
        test_size: Proporção do conjunto de teste (padrão 20%)
        
    Returns:
        Tuple com:
        - (X_train_cv, y_train_cv): 80% dos dados para rodar o Cross-Validation.
        - (X_test, y_test): 20% dos dados reservados para teste final.
    """
    df = load_toxicity_corpus()
    
    texts = df['Frase'].tolist()
    labels = df['Tox'].tolist()
    
    # Divisão: 80% Treino (para CV) vs 20% Teste (Final)
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"=== METODOLOGIA DE DIVISÃO ===")
    logger.info(f"Total de Amostras: {len(texts)}")
    logger.info(f"Conjunto de TREINO (Para Cross-Validation): {len(X_train_cv)} amostras ({(1-test_size)*100:.0f}%)")
    logger.info(f"Conjunto de TESTE (Reservado): {len(X_test)} amostras ({test_size*100:.0f}%)")
    
    return (X_train_cv, y_train_cv), (X_test, y_test)


if __name__ == "__main__":
    # Teste rápido para verificar carregamento
    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    print(f"\nDistribuição Treino: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Distribuição Teste: {pd.Series(y_test).value_counts().sort_index().to_dict()}")
