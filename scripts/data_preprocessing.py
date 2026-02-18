#!/usr/bin/env python3
"""
Script de pré-processamento de dados para modelos BERTimbau do YouTubeSafeKids.

Este script ajuda a preparar e limpar datasets para treinamento dos modelos,
incluindo limpeza de texto, balanceamento de classes e divisão train/test.

Uso:
    python data_preprocessing.py --input raw_data.csv --output processed_data.csv --model sentiment
    python data_preprocessing.py --input raw_data.csv --output processed_data.csv --model toxicity --balance
    python data_preprocessing.py --input raw_data.csv --output processed_data.csv --model language --clean-text
"""

import argparse
import logging
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
import unicodedata

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Pré-processar dados para modelos BERTimbau')
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Caminho para o arquivo de dados brutos'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Caminho para o arquivo de dados processados'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['sentiment', 'toxicity', 'language', 'educational'],
        help='Tipo de modelo para o qual preparar os dados'
    )
    
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Nome da coluna de texto (padrão: text)'
    )
    
    parser.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='Nome da coluna de rótulos (padrão: label)'
    )
    
    parser.add_argument(
        '--clean-text',
        action='store_true',
        help='Aplicar limpeza de texto'
    )
    
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balancear classes'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Dividir em train/test'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção para teste (padrão: 0.2)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Comprimento mínimo do texto (padrão: 10)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Comprimento máximo do texto (padrão: 512)'
    )
    
    parser.add_argument(
        '--remove-duplicates',
        action='store_true',
        help='Remover textos duplicados'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
    )
    
    return parser.parse_args()

def clean_text(text: str) -> str:
    """
    Limpa e normaliza texto.
    
    Args:
        text: Texto a ser limpo
        
    Returns:
        str: Texto limpo
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Normalizar unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remover caracteres de controle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar espaços em branco
    text = re.sub(r'\s+', ' ', text)
    
    # Remover URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remover menções e hashtags (opcional, dependendo do contexto)
    # text = re.sub(r'@\w+', '', text)
    # text = re.sub(r'#\w+', '', text)
    
    # Remover excesso de pontuação
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Limpar espaços extras
    text = text.strip()
    
    return text

def validate_labels(df: pd.DataFrame, model_type: str, label_column: str) -> pd.DataFrame:
    """
    Valida e normaliza rótulos baseado no tipo de modelo.
    
    Args:
        df: DataFrame com os dados
        model_type: Tipo do modelo
        label_column: Nome da coluna de rótulos
        
    Returns:
        pd.DataFrame: DataFrame com rótulos validados
    """
    logger.info(f"Validando rótulos para modelo {model_type}")
    
    # Mapeamentos de rótulos válidos
    valid_labels = {
        'sentiment': ['positive', 'negative', 'neutral'],
        'toxicity': ['toxic', 'non_toxic'],
        'language': ['appropriate', 'inappropriate'],
        'educational': ['educational', 'non_educational']
    }
    
    # Normalizar rótulos (lowercase, strip)
    df[label_column] = df[label_column].astype(str).str.lower().str.strip()
    
    # Mapear variações comuns
    label_mappings = {
        'sentiment': {
            'pos': 'positive', 'positivo': 'positive', '1': 'positive',
            'neg': 'negative', 'negativo': 'negative', '0': 'negative',
            'neu': 'neutral', 'neutro': 'neutral', '2': 'neutral'
        },
        'toxicity': {
            'tóxico': 'toxic', 'toxico': 'toxic', '1': 'toxic',
            'não tóxico': 'non_toxic', 'nao toxico': 'non_toxic', '0': 'non_toxic'
        },
        'language': {
            'nenhuma': 'appropriate', 'adequado': 'appropriate', '1': 'appropriate',
        'severa': 'inappropriate', 'inadequado': 'inappropriate', '0': 'inappropriate'
        },
        'educational': {
            'educativo': 'educational', 'educacional': 'educational', '1': 'educational',
            'não educativo': 'non_educational', 'nao educacional': 'non_educational', '0': 'non_educational'
        }
    }
    
    # Aplicar mapeamentos
    if model_type in label_mappings:
        df[label_column] = df[label_column].replace(label_mappings[model_type])
    
    # Filtrar apenas rótulos válidos
    valid_mask = df[label_column].isin(valid_labels[model_type])
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        logger.warning(f"Removendo {invalid_count} amostras com rótulos inválidos")
        logger.info(f"Rótulos únicos encontrados: {df[label_column].unique()}")
    
    df = df[valid_mask].copy()
    
    logger.info(f"Distribuição de rótulos: {df[label_column].value_counts().to_dict()}")
    
    return df

def balance_classes(df: pd.DataFrame, label_column: str, method: str = 'undersample') -> pd.DataFrame:
    """
    Balanceia classes no dataset.
    
    Args:
        df: DataFrame com os dados
        label_column: Nome da coluna de rótulos
        method: Método de balanceamento ('undersample' ou 'oversample')
        
    Returns:
        pd.DataFrame: DataFrame balanceado
    """
    logger.info(f"Balanceando classes usando método: {method}")
    
    # Contar amostras por classe
    class_counts = df[label_column].value_counts()
    logger.info(f"Distribuição original: {class_counts.to_dict()}")
    
    if method == 'undersample':
        # Undersample para a classe minoritária
        min_count = class_counts.min()
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            sampled_df = resample(class_df, n_samples=min_count, random_state=42)
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'oversample':
        # Oversample para a classe majoritária
        max_count = class_counts.max()
        
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            if len(class_df) < max_count:
                sampled_df = resample(class_df, n_samples=max_count, random_state=42, replace=True)
            else:
                sampled_df = class_df
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Embaralhar
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    new_counts = balanced_df[label_column].value_counts()
    logger.info(f"Distribuição balanceada: {new_counts.to_dict()}")
    
    return balanced_df

def filter_by_length(df: pd.DataFrame, text_column: str, min_length: int, max_length: int) -> pd.DataFrame:
    """
    Filtra textos por comprimento.
    
    Args:
        df: DataFrame com os dados
        text_column: Nome da coluna de texto
        min_length: Comprimento mínimo
        max_length: Comprimento máximo
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    logger.info(f"Filtrando textos por comprimento: {min_length}-{max_length} caracteres")
    
    # Calcular comprimentos
    text_lengths = df[text_column].str.len()
    
    # Aplicar filtros
    length_mask = (text_lengths >= min_length) & (text_lengths <= max_length)
    
    removed_count = (~length_mask).sum()
    if removed_count > 0:
        logger.info(f"Removendo {removed_count} textos fora do intervalo de comprimento")
    
    filtered_df = df[length_mask].copy()
    
    logger.info(f"Comprimento médio dos textos: {filtered_df[text_column].str.len().mean():.1f}")
    
    return filtered_df

def split_data(df: pd.DataFrame, test_size: float, label_column: str, seed: int = 42) -> tuple:
    """
    Divide dados em treino e teste.
    
    Args:
        df: DataFrame com os dados
        test_size: Proporção para teste
        label_column: Nome da coluna de rótulos
        seed: Seed para reprodutibilidade
        
    Returns:
        tuple: (train_df, test_df)
    """
    logger.info(f"Dividindo dados: {1-test_size:.1%} treino, {test_size:.1%} teste")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_column]
    )
    
    logger.info(f"Treino: {len(train_df)} amostras")
    logger.info(f"Teste: {len(test_df)} amostras")
    
    return train_df, test_df

def main():
    """Função principal do script de pré-processamento."""
    args = parse_arguments()
    
    logger.info("=== Iniciando Pré-processamento de Dados ===")
    logger.info(f"Arquivo de entrada: {args.input}")
    logger.info(f"Arquivo de saída: {args.output}")
    logger.info(f"Modelo: {args.model}")
    
    try:
        # Carregar dados
        logger.info("Carregando dados...")
        df = pd.read_csv(args.input)
        logger.info(f"Dados carregados: {len(df)} amostras")
        
        # Verificar colunas necessárias
        if args.text_column not in df.columns:
            raise ValueError(f"Coluna de texto '{args.text_column}' não encontrada")
        if args.label_column not in df.columns:
            raise ValueError(f"Coluna de rótulos '{args.label_column}' não encontrada")
        
        # Remover linhas com valores nulos
        initial_count = len(df)
        df = df.dropna(subset=[args.text_column, args.label_column])
        null_removed = initial_count - len(df)
        if null_removed > 0:
            logger.info(f"Removidas {null_removed} linhas com valores nulos")
        
        # Limpeza de texto
        if args.clean_text:
            logger.info("Aplicando limpeza de texto...")
            df[args.text_column] = df[args.text_column].apply(clean_text)
        
        # Filtrar por comprimento
        df = filter_by_length(df, args.text_column, args.min_length, args.max_length)
        
        # Remover duplicatas
        if args.remove_duplicates:
            initial_count = len(df)
            df = df.drop_duplicates(subset=[args.text_column])
            duplicates_removed = initial_count - len(df)
            if duplicates_removed > 0:
                logger.info(f"Removidas {duplicates_removed} duplicatas")
        
        # Validar rótulos
        df = validate_labels(df, args.model, args.label_column)
        
        # Balancear classes
        if args.balance:
            df = balance_classes(df, args.label_column)
        
        # Dividir dados se solicitado
        if args.split:
            train_df, test_df = split_data(df, args.test_size, args.label_column, args.seed)
            
            # Salvar arquivos separados
            base_path = Path(args.output)
            train_path = base_path.parent / f"{base_path.stem}_train{base_path.suffix}"
            test_path = base_path.parent / f"{base_path.stem}_test{base_path.suffix}"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Dados de treino salvos em: {train_path}")
            logger.info(f"Dados de teste salvos em: {test_path}")
        else:
            # Salvar arquivo único
            df.to_csv(args.output, index=False)
            logger.info(f"Dados processados salvos em: {args.output}")
        
        logger.info("=== Pré-processamento Concluído com Sucesso ===")
        logger.info(f"Total de amostras finais: {len(df)}")
        
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()