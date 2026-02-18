"""
Script de avaliação final para Detecção de Toxicidade (TOX).

Este script avalia o modelo treinado no conjunto de teste holdout (20%)
e gera relatórios detalhados com métricas e matriz de confusão.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.evaluation.model_evaluator import ModelEvaluator
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.preprocessing import preprocess_batch
from app.nlp.config import get_task_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Função principal de avaliação."""
    import argparse
    parser = argparse.ArgumentParser(description='Avaliação de Toxicidade')
    parser.add_argument('--preprocess', action='store_true',
                        help='Aplica pré-processamento (limpeza + lematização) se o modelo foi treinado assim')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Avaliação do Modelo de Detecção de Toxicidade")
    if args.preprocess:
        logger.info("PRÉ-PROCESSAMENTO: ATIVADO (Limpeza + Lematização)")
    logger.info("=" * 60)
    
    # Carrega conjunto de teste (holdout 20%)
    logger.info("Carregando conjunto de TESTE (Holdout 20%)...")
    _, (test_texts, test_labels) = get_data_for_cv_and_test()
    
    logger.info(f"Total de amostras de teste: {len(test_texts)}")
    logger.info(f"Distribuição: {pd.Series(test_labels).value_counts().sort_index().to_dict()}")
    
    # Aplica pré-processamento se solicitado
    if args.preprocess:
        logger.info("Aplicando pré-processamento aos dados de teste...")
        test_texts = preprocess_batch(test_texts, apply_cleaning=True, apply_lemmatization=True)
    
    # Configuração da tarefa
    task_config = get_task_config('TOX')
    class_names = task_config['classes']
    
    # Caminho do modelo treinado (busca o mais recente)
    models_dir = Path(__file__).parent.parent / 'models' / 'trained'
    
    # Lista modelos de toxicidade disponíveis
    model_paths = list(models_dir.glob('TOX_*'))
    if not model_paths:
        logger.error(f"Nenhum modelo TOX_* encontrado em {models_dir}")
        logger.error("Por favor, treine o modelo primeiro usando train_toxicity_gridsearch.py")
        return
    
    # Usa o modelo mais recente
    model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
    logger.info(f"Carregando modelo de: {model_path}")
    
    # Carrega modelo treinado
    model = BertimbauToxicity(model_path=str(model_path))
    
    # Cria avaliador
    output_dir = Path(__file__).parent / 'results' / 'toxicity_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(
        task_name='TOX',
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
        logger.info("Gerando relatório de avaliação e matriz de confusão...")
        evaluator.generate_report(results, model_path=str(model_path), save_plots=True)
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")
    
    # Imprime resultados
    logger.info("=" * 60)
    logger.info("Avaliação concluída!")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO (CONJUNTO HOLDOUT - 20%)")
    print("=" * 60)
    print(f"Acurácia: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"Precisão (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print("=" * 60)
    
    if 'classification_report' in results:
        print("\nRelatório de Classificação por Classe:")
        report_dict = results['classification_report']
        print(pd.DataFrame(report_dict).transpose())
    
    print(f"\nResultados salvos em: {output_dir}")
    print(f"Matriz de Confusão salva em: {output_dir}")
    
    # Informações para o artigo
    print("\n" + "=" * 60)
    print("DADOS PARA O ARTIGO CIENTÍFICO")
    print("=" * 60)
    print(f"Conjunto de Teste: {len(test_texts)} amostras")
    print(f"Classes: {class_names}")
    print(f"Modelo: BERTimbau Base (neuralmind/bert-base-portuguese-cased)")
    print(f"Caminho do modelo: {model_path}")


if __name__ == "__main__":
    main()
