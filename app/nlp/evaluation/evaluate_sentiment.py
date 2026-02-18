"""
Script de avaliação para Análise de Sentimentos (AS).

Este script avalia o modelo treinado de análise de sentimentos
no conjunto de teste e gera relatórios detalhados.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.evaluation.model_evaluator import ModelEvaluator
# CORREÇÃO: Importa a função correta criada para o Cross-Validation
from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
from app.nlp.config import get_task_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal de avaliação."""
    logger.info("=" * 60)
    logger.info("Iniciando avaliação do modelo de Análise de Sentimentos")
    logger.info("=" * 60)
    
    # Prepara dados de teste
    logger.info("Carregando conjunto de TESTE (Holdout 20%)...")
    
    # A função retorna ((treino_x, treino_y), (teste_x, teste_y))
    # Pegamos apenas a segunda parte (teste)
    _, (test_texts, test_labels) = get_data_for_cv_and_test()
    
    logger.info(f"Total de amostras de teste: {len(test_texts)}")
    
    # Configuração da tarefa
    task_config = get_task_config('AS')
    class_names = task_config['classes']
    
    # Caminho do modelo treinado
    # Por padrão, busca o modelo mais recente no diretório de modelos treinados
    models_dir = Path(__file__).parent.parent / 'models' / 'trained'
    
    # Lista modelos disponíveis
    model_paths = list(models_dir.glob('AS_*'))
    if not model_paths:
        logger.error(f"Nenhum modelo encontrado em {models_dir}")
        logger.error("Por favor, treine o modelo primeiro usando train_sentiment.py")
        return
    
    # Usa o modelo mais recente
    model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
    logger.info(f"Carregando modelo de: {model_path}")
    
    # Carrega modelo treinado
    model = BertimbauSentiment(model_path=str(model_path))
    
    # Cria avaliador
    output_dir = Path(__file__).parent / 'results' / 'sentiment_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(
        task_name='AS',
        class_names=class_names,
        output_dir=str(output_dir)
    )
    
    # Avalia modelo
    logger.info("Avaliando modelo no conjunto de teste...")
    results = evaluator.evaluate_model(
        model=model.model,
        tokenizer=model.tokenizer,
        test_texts=test_texts,
        test_labels=test_labels,
        max_length=model.max_length,
        batch_size=32
    )
    
    # Gera relatório e gráficos
    try:
        logger.info("Gerando relatório de avaliação e gráficos...")
        evaluator.generate_report(results, model_path=str(model_path), save_plots=True)
    except Exception as e:
        logger.error(f"Erro ao gerar relatório ou gráficos: {e}")
    
    logger.info("=" * 60)
    logger.info("Avaliação concluída!")
    logger.info("=" * 60)
    logger.info(f"Resultados salvos em: {output_dir}")
    
    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO (CONJUNTO HOLDOUT)")
    print("=" * 60)
    print(f"Acurácia: {results['accuracy']:.4f}")
    print(f"F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"Precisão (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print("=" * 60)
    
    if 'classification_report' in results:
        print("\nRelatório de Classificação:")
        report_dict = results['classification_report']
        print(pd.DataFrame(report_dict).transpose())
    
    print(f"\nResultados completos salvos em: {output_dir}")
    print(f"Gráfico da Matriz de Confusão salvo em: {output_dir}")


if __name__ == "__main__":
    main()