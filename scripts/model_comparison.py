#!/usr/bin/env python3
"""
Script de comparação de modelos BERTimbau do YouTubeSafeKids.

Este script permite comparar a performance de diferentes modelos, configurações
e versões, gerando relatórios detalhados e visualizações.

Uso:
    python model_comparison.py --models model1.pt model2.pt --test-data test.csv --output comparison_report
    python model_comparison.py --config comparison_config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, classification_report
)
import torch

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.nlp.bertimbau_sentiment import BertimbauSentiment
from app.models.nlp.bertimbau_toxicity import BertimbauToxicity
from app.models.nlp.bertimbau_language import BertimbauLanguage
from app.models.nlp.bertimbau_educational import BertimbauEducational

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar matplotlib para não usar GUI
plt.switch_backend('Agg')

def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Comparar modelos BERTimbau')
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Caminhos para os modelos a serem comparados'
    )
    
    parser.add_argument(
        '--model-names',
        nargs='+',
        help='Nomes dos modelos para exibição (opcional)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Caminho para os dados de teste'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['sentiment', 'toxicity', 'language', 'educational'],
        help='Tipo dos modelos'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model_comparison',
        help='Prefixo para arquivos de saída'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo de configuração JSON'
    )
    
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Nome da coluna de texto'
    )
    
    parser.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='Nome da coluna de rótulos'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch para avaliação'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Salvar predições individuais'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega configuração de arquivo JSON.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dict: Configuração carregada
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model(model_path: str, model_type: str):
    """
    Carrega modelo baseado no tipo.
    
    Args:
        model_path: Caminho para o modelo
        model_type: Tipo do modelo
        
    Returns:
        Modelo carregado
    """
    model_classes = {
        'sentiment': BertimbauSentiment,
        'toxicity': BertimbauToxicity,
        'language': BertimbauLanguage,
        'educational': BertimbauEducational
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    model_class = model_classes[model_type]
    model = model_class(model_path=model_path)
    
    return model

def evaluate_model(model, test_data: pd.DataFrame, text_column: str, 
                  label_column: str, model_type: str) -> Dict[str, Any]:
    """
    Avalia um modelo nos dados de teste.
    
    Args:
        model: Modelo a ser avaliado
        test_data: Dados de teste
        text_column: Nome da coluna de texto
        label_column: Nome da coluna de rótulos
        model_type: Tipo do modelo
        
    Returns:
        Dict: Métricas de avaliação
    """
    logger.info("Avaliando modelo...")
    
    texts = test_data[text_column].tolist()
    true_labels = test_data[label_column].tolist()
    
    # Fazer predições
    predictions = []
    confidences = []
    
    for text in texts:
        try:
            if model_type == 'sentiment':
                result = model.predict_sentiment(text)
                pred_class = result['class']
                confidence = result['confidence']
            elif model_type == 'toxicity':
                result = model.predict_toxicity(text)
                pred_class = 'toxic' if result['is_toxic'] else 'non_toxic'
                confidence = result['confidence']
            elif model_type == 'language':
                result = model.predict_language_appropriateness(text)
                pred_class = result['class']
                confidence = result['confidence']
            elif model_type == 'educational':
                result = model.predict_educational_value(text)
                pred_class = 'educational' if result['is_educational'] else 'non_educational'
                confidence = result['confidence']
            
            predictions.append(pred_class)
            confidences.append(confidence)
            
        except Exception as e:
            logger.warning(f"Erro na predição: {e}")
            predictions.append('unknown')
            confidences.append(0.0)
    
    # Calcular métricas
    metrics = calculate_metrics(true_labels, predictions, confidences, model_type)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'confidences': confidences,
        'true_labels': true_labels
    }

def calculate_metrics(true_labels: List[str], predictions: List[str], 
                     confidences: List[float], model_type: str) -> Dict[str, float]:
    """
    Calcula métricas de avaliação.
    
    Args:
        true_labels: Rótulos verdadeiros
        predictions: Predições do modelo
        confidences: Confianças das predições
        model_type: Tipo do modelo
        
    Returns:
        Dict: Métricas calculadas
    """
    # Métricas básicas
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': np.mean(confidences)
    }
    
    # Métricas específicas por tipo
    unique_labels = list(set(true_labels + predictions))
    
    if len(unique_labels) == 2:  # Classificação binária
        try:
            # Converter rótulos para numérico para AUC
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_true_numeric = [label_map.get(label, -1) for label in true_labels]
            y_pred_numeric = [label_map.get(pred, -1) for pred in predictions]
            
            # Filtrar predições inválidas
            valid_indices = [i for i, (true, pred) in enumerate(zip(y_true_numeric, y_pred_numeric)) 
                           if true != -1 and pred != -1]
            
            if valid_indices:
                y_true_valid = [y_true_numeric[i] for i in valid_indices]
                confidences_valid = [confidences[i] for i in valid_indices]
                
                auc = roc_auc_score(y_true_valid, confidences_valid)
                metrics['auc_roc'] = auc
        except Exception as e:
            logger.warning(f"Erro ao calcular AUC: {e}")
    
    # Métricas por classe
    class_report = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )
    
    for class_name, class_metrics in class_report.items():
        if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f'{class_name}_precision'] = class_metrics['precision']
            metrics[f'{class_name}_recall'] = class_metrics['recall']
            metrics[f'{class_name}_f1'] = class_metrics['f1-score']
    
    return metrics

def create_comparison_plots(results: Dict[str, Dict], output_prefix: str):
    """
    Cria gráficos de comparação entre modelos.
    
    Args:
        results: Resultados dos modelos
        output_prefix: Prefixo para arquivos de saída
    """
    logger.info("Criando gráficos de comparação...")
    
    # Preparar dados para visualização
    model_names = list(results.keys())
    metrics_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Gráfico de barras das métricas principais
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    df_main = df_metrics[df_metrics['Metric'].isin(main_metrics)]
    
    if not df_main.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_main, x='Metric', y='Value', hue='Model')
        plt.title('Comparação de Métricas Principais')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_main_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Matriz de confusão para cada modelo
    for model_name, result in results.items():
        true_labels = result['true_labels']
        predictions = result['predictions']
        
        # Calcular matriz de confusão
        labels = sorted(list(set(true_labels + predictions)))
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Rótulo Verdadeiro')
        plt.xlabel('Predição')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_confusion_matrix_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Gráfico de confiança
    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        confidences = result['confidences']
        plt.hist(confidences, alpha=0.7, label=model_name, bins=20)
    
    plt.title('Distribuição de Confiança das Predições')
    plt.xlabel('Confiança')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results: Dict[str, Dict], output_path: str):
    """
    Gera relatório detalhado da comparação.
    
    Args:
        results: Resultados dos modelos
        output_path: Caminho para o arquivo de relatório
    """
    logger.info("Gerando relatório...")
    
    report = {
        'summary': {},
        'detailed_results': results,
        'comparison': {}
    }
    
    # Resumo geral
    model_names = list(results.keys())
    report['summary']['models_compared'] = len(model_names)
    report['summary']['model_names'] = model_names
    
    # Encontrar melhor modelo por métrica
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    best_models = {}
    
    for metric in main_metrics:
        best_score = -1
        best_model = None
        
        for model_name, result in results.items():
            score = result['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        best_models[metric] = {
            'model': best_model,
            'score': best_score
        }
    
    report['comparison']['best_by_metric'] = best_models
    
    # Ranking geral (média das métricas principais)
    model_scores = {}
    for model_name, result in results.items():
        metrics = result['metrics']
        avg_score = np.mean([metrics.get(metric, 0) for metric in main_metrics])
        model_scores[model_name] = avg_score
    
    ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    report['comparison']['overall_ranking'] = ranking
    
    # Salvar relatório
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Criar versão em texto
    text_report_path = output_path.replace('.json', '.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE COMPARAÇÃO DE MODELOS ===\n\n")
        
        f.write(f"Modelos Comparados: {len(model_names)}\n")
        f.write(f"Nomes: {', '.join(model_names)}\n\n")
        
        f.write("=== RANKING GERAL ===\n")
        for i, (model, score) in enumerate(ranking, 1):
            f.write(f"{i}. {model}: {score:.4f}\n")
        f.write("\n")
        
        f.write("=== MELHOR MODELO POR MÉTRICA ===\n")
        for metric, info in best_models.items():
            f.write(f"{metric.upper()}: {info['model']} ({info['score']:.4f})\n")
        f.write("\n")
        
        f.write("=== MÉTRICAS DETALHADAS ===\n")
        for model_name, result in results.items():
            f.write(f"\n--- {model_name} ---\n")
            metrics = result['metrics']
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.4f}\n")

def main():
    """Função principal do script de comparação."""
    args = parse_arguments()
    
    logger.info("=== Iniciando Comparação de Modelos ===")
    
    try:
        # Carregar configuração se fornecida
        if args.config:
            config = load_config(args.config)
            # Sobrescrever argumentos com configuração
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        
        # Validar argumentos
        if not args.models:
            raise ValueError("Nenhum modelo especificado")
        
        # Carregar dados de teste
        logger.info(f"Carregando dados de teste: {args.test_data}")
        test_data = pd.read_csv(args.test_data)
        
        # Verificar colunas
        if args.text_column not in test_data.columns:
            raise ValueError(f"Coluna de texto '{args.text_column}' não encontrada")
        if args.label_column not in test_data.columns:
            raise ValueError(f"Coluna de rótulos '{args.label_column}' não encontrada")
        
        logger.info(f"Dados carregados: {len(test_data)} amostras")
        
        # Preparar nomes dos modelos
        if args.model_names:
            if len(args.model_names) != len(args.models):
                raise ValueError("Número de nomes deve corresponder ao número de modelos")
            model_names = args.model_names
        else:
            model_names = [f"Model_{i+1}" for i in range(len(args.models))]
        
        # Avaliar cada modelo
        results = {}
        
        for model_path, model_name in zip(args.models, model_names):
            logger.info(f"Avaliando modelo: {model_name} ({model_path})")
            
            try:
                # Carregar modelo
                model = load_model(model_path, args.model_type)
                
                # Avaliar
                result = evaluate_model(
                    model, test_data, args.text_column, 
                    args.label_column, args.model_type
                )
                
                results[model_name] = result
                
                logger.info(f"Modelo {model_name} avaliado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao avaliar modelo {model_name}: {e}")
                continue
        
        if not results:
            raise ValueError("Nenhum modelo foi avaliado com sucesso")
        
        # Gerar visualizações
        create_comparison_plots(results, args.output)
        
        # Gerar relatório
        report_path = f"{args.output}_report.json"
        generate_report(results, report_path)
        
        # Salvar predições se solicitado
        if args.save_predictions:
            predictions_path = f"{args.output}_predictions.csv"
            
            # Preparar DataFrame com todas as predições
            pred_data = test_data.copy()
            
            for model_name, result in results.items():
                pred_data[f'{model_name}_prediction'] = result['predictions']
                pred_data[f'{model_name}_confidence'] = result['confidences']
            
            pred_data.to_csv(predictions_path, index=False)
            logger.info(f"Predições salvas em: {predictions_path}")
        
        logger.info("=== Comparação Concluída com Sucesso ===")
        logger.info(f"Relatório salvo em: {report_path}")
        logger.info(f"Gráficos salvos com prefixo: {args.output}")
        
    except Exception as e:
        logger.error(f"Erro durante a comparação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()