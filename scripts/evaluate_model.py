#!/usr/bin/env python3
"""
Script de avaliação para modelos BERTimbau do YouTubeSafeKids.

Este script permite avaliar qualquer um dos 4 modelos específicos usando
métricas apropriadas para cada tipo de tarefa.

VERSÃO CORRIGIDA:
- Fix do erro KeyError: 'class' -> 'predicted_class'
- Salva o texto original no CSV de predições
- Exibe relatório detalhado por classe no console
"""

import argparse
import logging
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.models.bertimbau_language import BertimbauLanguage
from app.nlp.models.bertimbau_educational import BertimbauEducational

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    parser = argparse.ArgumentParser(description='Avaliar modelos BERTimbau')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['sentiment', 'toxicity', 'language', 'educational'],
        help='Tipo de modelo para avaliar'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Caminho para o arquivo de dados de teste'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Caminho para o modelo treinado'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Diretório de saída para os resultados'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch para avaliação (padrão: 32)'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Salvar predições individuais'
    )
    
    return parser.parse_args()

def load_test_data(data_path: str, model_type: str) -> pd.DataFrame:
    """Carrega dados de teste baseado no tipo de modelo."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validar colunas necessárias
    required_columns = {
        'sentiment': ['text', 'label'],
        'toxicity': ['text', 'label'],
        'language': ['text', 'label'],
        'educational': ['text', 'label']
    }
    
    missing_columns = set(required_columns[model_type]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
    
    logger.info(f"Dados carregados: {len(df)} amostras")
    return df

def evaluate_sentiment_model(model, test_data: pd.DataFrame) -> dict:
    """Avalia modelo de sentimento."""
    logger.info("Avaliando modelo de sentimento...")
    
    predictions = []
    true_labels = []
    texts = [] # Novo: Guardar textos
    confidences = [] # Novo: Guardar confiança
    
    # Mapeamento para relatório bonito (ajuste se seus labels forem diferentes)
    target_names = ['Negativo', 'Neutro', 'Positivo']
    
    for _, row in test_data.iterrows():
        try:
            result = model.predict_sentiment(row['text'])
            
            # --- FIX: Usar predicted_class ---
            pred_class = result.get('predicted_class', result.get('class'))
            predictions.append(pred_class)
            true_labels.append(row['label'])
            texts.append(row['text'])
            confidences.append(result.get('confidence', 0.0))
            
        except Exception as e:
            logger.warning(f"Erro ao processar amostra: {e}")
            continue
    
    # Calcular métricas
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    # Relatório formatado para o console
    cls_report_str = classification_report(true_labels, predictions, target_names=target_names, digits=4, zero_division=0)
    cls_report_dict = classification_report(true_labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
    
    print("\n" + "="*60)
    print("RELATÓRIO DETALHADO POR CLASSE (SENTIMENTO)")
    print("="*60)
    print(cls_report_str)
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f1_score_macro': f1_macro,
        'classification_report': cls_report_dict,
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'predictions': predictions,
        'true_labels': true_labels,
        'texts': texts,
        'confidences': confidences
    }

def evaluate_toxicity_model(model, test_data: pd.DataFrame) -> dict:
    """Avalia modelo de toxicidade."""
    logger.info("Avaliando modelo de toxicidade...")
    
    predictions = []
    confidences = []
    true_labels = []
    texts = []
    
    for _, row in test_data.iterrows():
        try:
            result = model.predict_toxicity(row['text'])
            # --- FIX: Usar predicted_class ---
            pred_class = result.get('predicted_class', result.get('class'))
            
            predictions.append(pred_class)
            confidences.append(result['confidence'])
            true_labels.append(row['label'])
            texts.append(row['text'])
        except Exception as e:
            logger.warning(f"Erro ao processar amostra: {e}")
            continue
    
    # Calcular métricas
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary', pos_label='TOXIC', zero_division=0)
    recall = recall_score(true_labels, predictions, average='binary', pos_label='TOXIC', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='binary', pos_label='TOXIC', zero_division=0)
    
    auc_roc = None
    try:
        y_true_binary = [1 if label == 'TOXIC' else 0 for label in true_labels]
        # Ajuste simples para probabilidade
        y_pred_proba = [conf if pred == 'TOXIC' else 1-conf for pred, conf in zip(predictions, confidences)]
        auc_roc = roc_auc_score(y_true_binary, y_pred_proba)
    except Exception as e:
        logger.warning(f"Não foi possível calcular AUC-ROC: {e}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'classification_report': classification_report(true_labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'predictions': predictions,
        'confidences': confidences,
        'true_labels': true_labels,
        'texts': texts
    }

def evaluate_language_model(model, test_data: pd.DataFrame) -> dict:
    """Avalia modelo de linguagem imprópria."""
    logger.info("Avaliando modelo de linguagem...")
    
    predictions = []
    true_labels = []
    texts = []
    
    for _, row in test_data.iterrows():
        try:
            result = model.predict_language_appropriateness(row['text'])
            # --- FIX: Usar predicted_class ---
            pred_class = result.get('predicted_class', result.get('class'))
            
            predictions.append(pred_class)
            true_labels.append(row['label'])
            texts.append(row['text'])
        except Exception as e:
            logger.warning(f"Erro ao processar amostra: {e}")
            continue
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary', pos_label='INAPPROPRIATE', zero_division=0)
    recall = recall_score(true_labels, predictions, average='binary', pos_label='INAPPROPRIATE', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='binary', pos_label='INAPPROPRIATE', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(true_labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'predictions': predictions,
        'true_labels': true_labels,
        'texts': texts
    }

def evaluate_educational_model(model, test_data: pd.DataFrame) -> dict:
    """Avalia modelo educacional."""
    logger.info("Avaliando modelo educacional...")
    
    predictions = []
    scores = []
    true_labels = []
    texts = []
    
    for _, row in test_data.iterrows():
        try:
            result = model.predict_educational_value(row['text'])
            score = model.get_educational_score(result)
            
            # --- FIX: Lógica de classe ---
            if 'predicted_class' in result:
                pred_class = result['predicted_class']
            elif 'class' in result:
                pred_class = result['class']
            else:
                pred_class = 'EDUCATIONAL' if score > 0.5 else 'NON_EDUCATIONAL'
            
            predictions.append(pred_class)
            scores.append(score)
            true_labels.append(row['label'])
            texts.append(row['text'])
        except Exception as e:
            logger.warning(f"Erro ao processar amostra: {e}")
            continue
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    mean_score = np.mean(scores) if scores else 0
    std_score = np.std(scores) if scores else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_educational_score': mean_score,
        'std_educational_score': std_score,
        'classification_report': classification_report(true_labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'predictions': predictions,
        'scores': scores,
        'true_labels': true_labels,
        'texts': texts
    }

def save_results(results: dict, output_dir: str, model_type: str, save_predictions: bool = False):
    """Salva resultados da avaliação."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar métricas principais (JSON)
    metrics_file = os.path.join(output_dir, f"{model_type}_metrics_{timestamp}.json")
    
    json_results = {}
    # Ignora listas longas no JSON de métricas
    exclude_keys = ['predictions', 'true_labels', 'confidences', 'scores', 'texts']
    
    for key, value in results.items():
        if key in exclude_keys:
            continue
        elif isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Métricas salvas em: {metrics_file}")
    
    # Salvar predições detalhadas (CSV) se solicitado
    if save_predictions and 'predictions' in results:
        predictions_file = os.path.join(output_dir, f"{model_type}_predictions_{timestamp}.csv")
        
        data_dict = {
            'true_label': results['true_labels'],
            'predicted_label': results['predictions']
        }
        
        # Adiciona o texto se disponível (Melhoria solicitada)
        if 'texts' in results and len(results['texts']) == len(results['predictions']):
            data_dict['text'] = results['texts']
            # Reordenar para o texto ficar primeiro
            cols = ['text', 'true_label', 'predicted_label']
        else:
            cols = ['true_label', 'predicted_label']

        if 'confidences' in results:
            data_dict['confidence'] = results['confidences']
            cols.append('confidence')
            
        if 'scores' in results:
            data_dict['score'] = results['scores']
            cols.append('score')
        
        pred_df = pd.DataFrame(data_dict)
        
        # Garante a ordem das colunas
        final_cols = [c for c in cols if c in pred_df.columns]
        pred_df = pred_df[final_cols]
        
        # Salvar com ; para facilitar abertura no Excel BR
        pred_df.to_csv(predictions_file, index=False, sep=';', encoding='utf-8-sig')
        logger.info(f"Predições salvas em: {predictions_file}")

def main():
    """Função principal do script de avaliação."""
    args = parse_arguments()
    
    logger.info("=== Iniciando Avaliação de Modelo BERTimbau ===")
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Dados: {args.data}")
    logger.info(f"Modelo Path: {args.model_path}")
    
    try:
        # Carregar dados de teste
        test_data = load_test_data(args.data, args.model)
        
        # Carregar modelo
        model_class = MODELS[args.model]
        model = model_class(args.model_path)
        
        logger.info("Modelo carregado com sucesso")
        
        # Avaliar modelo baseado no tipo
        if args.model == 'sentiment':
            results = evaluate_sentiment_model(model, test_data)
        elif args.model == 'toxicity':
            results = evaluate_toxicity_model(model, test_data)
        elif args.model == 'language':
            results = evaluate_language_model(model, test_data)
        elif args.model == 'educational':
            results = evaluate_educational_model(model, test_data)
        
        # Exibir resultados principais no log
        logger.info("=== Resultados Gerais ===")
        logger.info(f"Acurácia: {results['accuracy']:.4f}")
        logger.info(f"Precisão (Weighted): {results['precision']:.4f}")
        logger.info(f"Recall (Weighted): {results['recall']:.4f}")
        logger.info(f"F1-Score (Weighted): {results['f1_score']:.4f}")
        
        if 'f1_score_macro' in results:
             logger.info(f"F1-Score (Macro): {results['f1_score_macro']:.4f}")
        
        if 'auc_roc' in results and results['auc_roc']:
            logger.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        
        if 'mean_educational_score' in results:
            logger.info(f"Score Educacional Médio: {results['mean_educational_score']:.4f}")
        
        # Salvar resultados
        save_results(results, args.output, args.model, args.save_predictions)
        
        logger.info("=== Avaliação Concluída com Sucesso ===")
        
    except Exception as e:
        logger.error(f"Erro durante a avaliação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()