#!/usr/bin/env python3
"""
Script de treinamento para modelos BERTimbau do YouTubeSafeKids.

Este script permite treinar qualquer um dos 4 modelos específicos:
- Sentimento (AS)
- Toxicidade (TO) 
- Linguagem Imprópria (LI)
- Tópicos Educacionais (TE)

Uso:
    python train_model.py --model sentiment --data data/sentiment_dataset.csv
    python train_model.py --model toxicity --data data/toxicity_dataset.csv --epochs 5
    python train_model.py --model language --data data/language_dataset.csv --batch-size 16
    python train_model.py --model educational --data data/educational_dataset.csv --lr 2e-5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.models.bertimbau_language import BertimbauLanguage
from app.nlp.models.bertimbau_educational import BertimbauEducational

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mapeamento de modelos
MODELS = {
    'sentiment': BertimbauSentiment,
    'toxicity': BertimbauToxicity,
    'language': BertimbauLanguage,
    'educational': BertimbauEducational
}

def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Treinar modelos BERTimbau')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['sentiment', 'toxicity', 'language', 'educational'],
        help='Tipo de modelo para treinar'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Caminho para o arquivo de dados de treinamento'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/trained',
        help='Diretório de saída para o modelo treinado'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Número de épocas de treinamento (padrão: 3)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Tamanho do batch (padrão: 8)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Taxa de aprendizado (padrão: 2e-5)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Comprimento máximo das sequências (padrão: 512)'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Proporção dos dados para validação (padrão: 0.2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Caminho para modelo pré-treinado para continuar o treinamento'
    )
    
    return parser.parse_args()

def validate_data_file(data_path: str, model_type: str) -> bool:
    """
    Valida se o arquivo de dados existe e tem o formato correto.
    
    Args:
        data_path: Caminho para o arquivo de dados
        model_type: Tipo do modelo (sentiment, toxicity, etc.)
        
    Returns:
        bool: True se válido, False caso contrário
    """
    if not os.path.exists(data_path):
        logger.error(f"Arquivo de dados não encontrado: {data_path}")
        return False
    
    # TODO: Implementar validação específica do formato do dataset
    # Cada modelo pode ter colunas diferentes:
    # - sentiment: text, label (positive/negative/neutral)
    # - toxicity: text, label (toxic/non_toxic)
    # - language: text, label (appropriate/inappropriate)
    # - educational: text, label, age_group, topic
    
    logger.info(f"Arquivo de dados validado: {data_path}")
    return True

def setup_output_directory(output_path: str, model_type: str) -> str:
    """
    Configura o diretório de saída para o modelo.
    
    Args:
        output_path: Diretório base de saída
        model_type: Tipo do modelo
        
    Returns:
        str: Caminho completo para o diretório do modelo
    """
    model_dir = os.path.join(output_path, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Diretório de saída configurado: {model_dir}")
    return model_dir

def main():
    """Função principal do script de treinamento."""
    args = parse_arguments()
    
    logger.info("=== Iniciando Treinamento de Modelo BERTimbau ===")
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Dados: {args.data}")
    logger.info(f"Épocas: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    
    # Validar arquivo de dados
    if not validate_data_file(args.data, args.model):
        sys.exit(1)
    
    # Configurar diretório de saída
    output_dir = setup_output_directory(args.output, args.model)
    
    try:
        # Inicializar modelo
        model_class = MODELS[args.model]
        
        if args.resume:
            logger.info(f"Carregando modelo pré-treinado: {args.resume}")
            model = model_class(args.resume)
        else:
            logger.info("Inicializando novo modelo")
            model = model_class()
        
        # Configurar parâmetros de treinamento
        training_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_length': args.max_length,
            'validation_split': args.validation_split,
            'seed': args.seed,
            'output_dir': output_dir
        }
        
        logger.info("Iniciando treinamento...")
        
        # Treinar modelo
        model.train_model(
            data_path=args.data,
            **training_config
        )
        
        logger.info("=== Treinamento Concluído com Sucesso ===")
        logger.info(f"Modelo salvo em: {output_dir}")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()