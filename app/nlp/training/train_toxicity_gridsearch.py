"""
Script de treinamento com Grid Search para Detecção de Toxicidade.

Metodologia Científica:
- Grid Search com Successive Halving (2 estágios)
- Cross-Validation (K=5) para cada configuração
- Oversampling apenas no conjunto de treino de cada fold
- Métricas: F1 Macro, F1 Weighted, Precision, Recall
- Holdout de 20% para teste final (nunca usado no tuning)

Modos de execução:
- FAST: Grid Search reduzido (1 configuração) para validar pipeline
- FULL: Grid Search completo (72 configurações) - uso não recomendado
- OPTIMIZED: Grid Search com Successive Halving (recomendado para artigo)
  - Estágio 1: Todas configs × 2 folds × 3 épocas (triagem rápida)
  - Estágio 2: Top-N configs × 5 folds × épocas completas (avaliação final)

Opções de pré-processamento:
- --preprocess: Aplica limpeza de texto e lematização (conforme TCC)
- Sem flag: Usa texto original (apenas tokenização do BERT)
"""

import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from pathlib import Path
from sklearn.utils import resample
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split
from app.nlp.utils.preprocessing import preprocess_batch
from app.nlp.config import PATHS, get_task_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES DO GRID SEARCH
# ============================================================================

# MODO FAST: Grid Search reduzido para validação rápida do pipeline
GRID_FAST = {
    'epochs': [3],
    'batch_size': [16],
    'learning_rate': [5e-5],
    'max_length': [128]
}

# MODO FULL: Grid Search ampliado original (NÃO RECOMENDADO - ~56 dias)
GRID_FULL = {
    'epochs': [3, 5, 8],
    'batch_size': [4, 8, 16],
    'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
    'max_length': [128, 256]
}

# MODO OPTIMIZED: Grid cientificamente otimizado (base para Successive Halving)
# Justificativa: Devlin et al. (2019) recomendam batch_size ∈ {8, 16, 32},
# lr ∈ {2e-5, 3e-5, 5e-5}, epochs ∈ {2, 3, 4} para fine-tuning de BERT.
# max_length=128 é suficiente para comentários curtos do YouTube.
GRID_OPTIMIZED = {
    'epochs': [3, 5],
    'batch_size': [8, 16],
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'max_length': [128]
}

# Configurações para Successive Halving (modo optimized)
STAGE1_FOLDS = 2          # Folds reduzidos para triagem rápida
STAGE1_EPOCHS = 3         # Épocas fixas no estágio 1 (apenas para ranking)
TOP_N_CONFIGS = 6         # Quantas configs avançam para o estágio 2

K_FOLDS = 5

# Mapeamento de parâmetros do Grid para argumentos do Trainer
GRID_PARAM_MAPPING = {
    'epochs': 'num_train_epochs',
    'batch_size': 'per_device_train_batch_size',
    'learning_rate': 'learning_rate',
    'max_length': 'max_length'
}


def apply_oversampling(train_texts, train_labels):
    """
    Balanceia classes minoritárias via Random Oversampling.
    Aplicado APENAS no conjunto de treino para evitar data leakage.
    
    Classes: 0=Nenhuma, 1=Leve, 2=Severa
    """
    df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Separa por classe
    df_class_0 = df[df['label'] == 0]  # Nenhuma
    df_class_1 = df[df['label'] == 1]  # Leve
    df_class_2 = df[df['label'] == 2]  # Severa
    
    # Define alvo como a classe majoritária
    max_count = max(len(df_class_0), len(df_class_1), len(df_class_2))
    
    # Upsample das classes minoritárias
    df_0_up = resample(df_class_0, replace=True, n_samples=max_count, random_state=42)
    df_1_up = resample(df_class_1, replace=True, n_samples=max_count, random_state=42)
    df_2_up = resample(df_class_2, replace=True, n_samples=max_count, random_state=42)
    
    df_balanced = pd.concat([df_0_up, df_1_up, df_2_up])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()


def _train_single_fold(config, X_train_cv, y_train_cv, train_idx, val_idx,
                       config_idx, fold, total_folds, override_epochs=None):
    """
    Treina um único fold de uma configuração.
    
    Args:
        config: Dict com os hiperparâmetros
        X_train_cv: Array com todos os textos de CV
        y_train_cv: Array com todos os labels de CV
        train_idx: Índices do treino neste fold
        val_idx: Índices da validação neste fold
        config_idx: Índice da configuração (para logging)
        fold: Número do fold atual (0-indexed)
        total_folds: Total de folds
        override_epochs: Se fornecido, sobrescreve o número de épocas da config
        
    Returns:
        Dict com métricas do fold ou None em caso de erro
    """
    curr_fold = fold + 1
    logger.info(f"\n>>> [Fold {curr_fold}/{total_folds}]")
    
    # Separação do fold
    X_fold_train = X_train_cv[train_idx]
    X_fold_val = X_train_cv[val_idx]
    y_fold_train = y_train_cv[train_idx]
    y_fold_val = y_train_cv[val_idx]
    
    # Oversampling apenas no treino
    X_train_bal, y_train_bal = apply_oversampling(
        X_fold_train.tolist(), 
        y_fold_train.tolist()
    )
    
    logger.info(f"  Treino (balanceado): {len(X_train_bal)} | Validação: {len(X_fold_val)}")
    
    # Instancia modelo limpo
    model = BertimbauToxicity()
    
    # Prepara kwargs com hiperparâmetros
    train_kwargs = {}
    effective_config = config.copy()
    if override_epochs is not None:
        effective_config['epochs'] = override_epochs
        
    for p_name, p_val in effective_config.items():
        if p_name in GRID_PARAM_MAPPING:
            target_arg = GRID_PARAM_MAPPING[p_name]
            train_kwargs[target_arg] = p_val
            # Ajusta batch size de validação também
            if p_name == 'batch_size':
                train_kwargs['per_device_eval_batch_size'] = p_val

    # Treina o modelo com os parâmetros da configuração atual
    try:
        epoch_label = f"ep={effective_config['epochs']}"
        results = model.train_model(
            train_texts=X_train_bal,
            train_labels=y_train_bal,
            val_texts=X_fold_val.tolist(),
            val_labels=y_fold_val.tolist(),
            config_name='default',
            experiment_name=f'TOX_grid_{config_idx}_fold_{curr_fold}_{epoch_label}',
            save_artifacts=False,
            loss_type='focal',
            focal_gamma=2.0,
            label_smoothing=0.1,
            **train_kwargs
        )
        
        metrics = results.get('final_metrics', {})
        metrics['fold'] = curr_fold
        
        f1_val = metrics.get('eval_f1', metrics.get('f1', 0))
        logger.info(f"  [Fold {curr_fold}] F1: {f1_val:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"  Erro no fold {curr_fold}: {e}")
        return {'fold': curr_fold, 'error': str(e)}


def _evaluate_configs(combinations, param_names, X_train_cv, y_train_cv,
                      n_folds, override_epochs=None, stage_label=""):
    """
    Avalia uma lista de configurações com cross-validation.
    
    Args:
        combinations: Lista de tuplas com valores dos hiperparâmetros
        param_names: Nomes dos hiperparâmetros
        X_train_cv: Array de textos
        y_train_cv: Array de labels
        n_folds: Número de folds para cross-validation
        override_epochs: Se fornecido, sobrescreve épocas em todas as configs
        stage_label: Label para logging (ex: "ESTÁGIO 1", "ESTÁGIO 2")
        
    Returns:
        Lista de dicts com resultados por configuração
    """
    all_results = []
    
    for config_idx, params in enumerate(combinations, 1):
        config = dict(zip(param_names, params))
        logger.info(f"\n{'='*70}")
        if stage_label:
            logger.info(f"{stage_label} — CONFIGURAÇÃO {config_idx}/{len(combinations)}")
        else:
            logger.info(f"CONFIGURAÇÃO {config_idx}/{len(combinations)}")
        
        epoch_display = override_epochs if override_epochs else config.get('epochs', '?')
        logger.info(f"Parâmetros: {config} (épocas efetivas: {epoch_display})")
        logger.info(f"{'='*70}")
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(
            get_kfold_split(X_train_cv, y_train_cv, n_splits=n_folds)
        ):
            metrics = _train_single_fold(
                config, X_train_cv, y_train_cv,
                train_idx, val_idx,
                config_idx, fold, n_folds,
                override_epochs=override_epochs
            )
            fold_metrics.append(metrics)
        
        # Calcula métricas médias da configuração
        valid_metrics = [m for m in fold_metrics if 'error' not in m]
        if valid_metrics:
            df_metrics = pd.DataFrame(valid_metrics)
            
            config_result = {
                'config': config,
                'config_idx': config_idx,
                'n_folds_ok': len(valid_metrics),
                'mean_f1': df_metrics['eval_f1'].mean() if 'eval_f1' in df_metrics else 0,
                'std_f1': df_metrics['eval_f1'].std() if 'eval_f1' in df_metrics else 0,
                'mean_accuracy': df_metrics['eval_accuracy'].mean() if 'eval_accuracy' in df_metrics else 0,
                'mean_loss': df_metrics['eval_loss'].mean() if 'eval_loss' in df_metrics else 0,
                'fold_metrics': fold_metrics
            }
            
            all_results.append(config_result)
            
            logger.info(f"\nMédia Config {config_idx}: "
                        f"F1={config_result['mean_f1']:.4f} "
                        f"(+/- {config_result['std_f1']:.4f})")
        else:
            logger.warning(f"Configuração {config_idx} falhou em todos os folds!")
    
    return all_results


def run_grid_search_optimized(preprocess=False):
    """
    Executa Grid Search Otimizado com Successive Halving (2 estágios).
    
    Estágio 1: Triagem rápida (2 folds, 3 épocas) de todas as configurações
    Estágio 2: Avaliação completa (5 folds, épocas originais) das Top-N configs
    
    Referência metodológica:
    - Jamieson & Talwalkar (2016). "Non-stochastic Best Arm Identification 
      and Hyperparameter Optimization" (Successive Halving)
    
    Args:
        preprocess: Se True, aplica limpeza + lematização nos textos
    """
    grid = GRID_OPTIMIZED
    preprocess_label = "COM" if preprocess else "SEM"
    
    logger.info("=" * 70)
    logger.info("GRID SEARCH OTIMIZADO (SUCCESSIVE HALVING) — TOXICIDADE")
    logger.info(f"PRÉ-PROCESSAMENTO: {preprocess_label} (limpeza + lematização)")
    logger.info("=" * 70)
    logger.info(f"Grid base: {grid}")
    
    # Gera todas as combinações
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Total de configurações: {len(combinations)}")
    logger.info(f"Estágio 1: {len(combinations)} configs × {STAGE1_FOLDS} folds × {STAGE1_EPOCHS} épocas")
    logger.info(f"Estágio 2: Top-{TOP_N_CONFIGS} configs × {K_FOLDS} folds × épocas originais")
    stage1_treinos = len(combinations) * STAGE1_FOLDS
    stage2_treinos = TOP_N_CONFIGS * K_FOLDS
    logger.info(f"Total estimado de treinos: {stage1_treinos} + {stage2_treinos} = {stage1_treinos + stage2_treinos}")
    
    # Carrega dados (80% CV / 20% Teste)
    (X_train_cv_list, y_train_cv_list), (X_test_final, y_test_final) = get_data_for_cv_and_test(test_size=0.20)
    
    # Aplica pré-processamento se solicitado
    if preprocess:
        logger.info("Aplicando pré-processamento (limpeza + lematização)...")
        X_train_cv_list = preprocess_batch(X_train_cv_list, apply_cleaning=True, apply_lemmatization=True)
        X_test_final = preprocess_batch(X_test_final, apply_cleaning=True, apply_lemmatization=True, show_progress=False)
    
    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    
    # ===========================================================================
    # ESTÁGIO 1: Triagem Rápida
    # ===========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTÁGIO 1: TRIAGEM RÁPIDA")
    logger.info(f"Configurações: {len(combinations)} | Folds: {STAGE1_FOLDS} | Épocas: {STAGE1_EPOCHS}")
    logger.info("=" * 70)
    
    stage1_results = _evaluate_configs(
        combinations, param_names, X_train_cv, y_train_cv,
        n_folds=STAGE1_FOLDS,
        override_epochs=STAGE1_EPOCHS,
        stage_label="ESTÁGIO 1"
    )
    
    # Ranking por F1 médio
    stage1_results.sort(key=lambda x: x['mean_f1'], reverse=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("RANKING DO ESTÁGIO 1 (TRIAGEM)")
    logger.info("=" * 70)
    for rank, result in enumerate(stage1_results, 1):
        marker = " ✓ AVANÇA" if rank <= TOP_N_CONFIGS else ""
        logger.info(
            f"  #{rank}: F1={result['mean_f1']:.4f} "
            f"(+/- {result['std_f1']:.4f}) | "
            f"{result['config']}{marker}"
        )
    
    # Seleciona Top-N
    top_configs = stage1_results[:TOP_N_CONFIGS]
    top_combinations = [tuple(c['config'][p] for p in param_names) for c in top_configs]
    
    logger.info(f"\n{TOP_N_CONFIGS} configurações selecionadas para o Estágio 2.")
    
    # ===========================================================================
    # ESTÁGIO 2: Avaliação Completa
    # ===========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTÁGIO 2: AVALIAÇÃO COMPLETA")
    logger.info(f"Configurações: {len(top_combinations)} | Folds: {K_FOLDS} | Épocas: originais da config")
    logger.info("=" * 70)
    
    stage2_results = _evaluate_configs(
        top_combinations, param_names, X_train_cv, y_train_cv,
        n_folds=K_FOLDS,
        override_epochs=None,  # Usa as épocas originais da config
        stage_label="ESTÁGIO 2"
    )
    
    # Ranking final
    stage2_results.sort(key=lambda x: x['mean_f1'], reverse=True)
    
    # Identifica melhor configuração
    best_config = stage2_results[0] if stage2_results else None
    
    # Relatório Final
    _print_final_report(stage1_results, stage2_results, best_config)
    
    # Salva resultados
    output_dir = Path(__file__).parent.parent / 'evaluation' / 'results' / 'toxicity_gridsearch'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocess_suffix = "_preprocess" if preprocess else "_raw"
    results_file = output_dir / f'gridsearch_optimized{preprocess_suffix}_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'optimized',
            'method': 'successive_halving',
            'preprocessing': preprocess,
            'preprocessing_description': 'limpeza + lematização' if preprocess else 'texto original',
            'grid': grid,
            'stage1': {
                'folds': STAGE1_FOLDS,
                'epochs_override': STAGE1_EPOCHS,
                'total_configs': len(combinations),
                'results': stage1_results
            },
            'stage2': {
                'folds': K_FOLDS,
                'top_n': TOP_N_CONFIGS,
                'results': stage2_results
            },
            'best_config': best_config,
            'test_set_size': len(X_test_final)
        }, f, indent=2, default=str)
    
    logger.info(f"\nResultados salvos em: {results_file}")
    
    # Treina modelo final com a melhor config
    best_model_path = None
    if best_config:
        try:
            train_final_best_model(best_config, X_train_cv, y_train_cv, X_test_final, y_test_final)
            models_dir = Path(__file__).parent.parent / 'models' / 'trained'
            candidates = list(models_dir.glob('TOX_*'))
            if candidates:
                best_model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        except Exception as e:
            logger.error(f"Erro ao treinar e localizar modelo final: {e}")
    
    return stage2_results, best_config, best_model_path


def _print_final_report(stage1_results, stage2_results, best_config):
    """Imprime relatório final consolidado."""
    print("\n" + "=" * 70)
    print("RESUMO FINAL DO GRID SEARCH OTIMIZADO (SUCCESSIVE HALVING)")
    print("=" * 70)
    
    print(f"\n--- Estágio 1 (Triagem): {len(stage1_results)} configs avaliadas ---")
    stage1_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
        }
        for r in stage1_results
    ])
    print(stage1_df.to_string(index=False))
    
    print(f"\n--- Estágio 2 (Avaliação Completa): Top-{TOP_N_CONFIGS} configs ---")
    stage2_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
            'Acc_mean': r['mean_accuracy']
        }
        for r in stage2_results
    ])
    print(stage2_df.to_string(index=False))
    
    if best_config:
        print(f"\n{'='*70}")
        print("MELHOR CONFIGURAÇÃO (VENCEDORA)")
        print(f"{'='*70}")
        print(f"Parâmetros: {best_config['config']}")
        print(f"F1-Score Médio: {best_config['mean_f1']:.4f} (+/- {best_config['std_f1']:.4f})")
        print(f"Accuracy Média: {best_config['mean_accuracy']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"OBSERVAÇÃO IMPORTANTE:")
    print(f"Use evaluate_toxicity.py para avaliar o modelo final no conjunto de teste.")
    print(f"{'='*70}")


def run_grid_search(mode='fast', preprocess=False):
    """
    Executa Grid Search com Cross-Validation.
    
    Args:
        mode: 'fast' para validação rápida, 'full' para grid completo,
              'optimized' para successive halving (recomendado)
        preprocess: Se True, aplica limpeza + lematização nos textos
    """
    # Redireciona para o modo otimizado
    if mode == 'optimized':
        return run_grid_search_optimized(preprocess=preprocess)
    
    grid = GRID_FAST if mode == 'fast' else GRID_FULL
    preprocess_label = "COM" if preprocess else "SEM"
    
    logger.info("=" * 70)
    logger.info(f"INICIANDO GRID SEARCH ({mode.upper()}) - TOXICIDADE")
    logger.info(f"PRÉ-PROCESSAMENTO: {preprocess_label} (limpeza + lematização)")
    logger.info("=" * 70)
    logger.info(f"Hiperparâmetros a testar: {grid}")
    
    # Gera todas as combinações
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Total de configurações: {len(combinations)}")
    logger.info(f"Folds por configuração: {K_FOLDS}")
    logger.info(f"Total de treinos: {len(combinations) * K_FOLDS}")
    
    # Carrega dados (80% CV / 20% Teste)
    (X_train_cv_list, y_train_cv_list), (X_test_final, y_test_final) = get_data_for_cv_and_test(test_size=0.20)
    
    # Aplica pré-processamento se solicitado
    if preprocess:
        logger.info("Aplicando pré-processamento (limpeza + lematização)...")
        X_train_cv_list = preprocess_batch(X_train_cv_list, apply_cleaning=True, apply_lemmatization=True)
        X_test_final = preprocess_batch(X_test_final, apply_cleaning=True, apply_lemmatization=True, show_progress=False)
    
    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    
    all_results = _evaluate_configs(
        combinations, param_names, X_train_cv, y_train_cv,
        n_folds=K_FOLDS
    )
    
    # Identifica melhor configuração
    best_config = None
    best_f1_macro = 0.0
    for result in all_results:
        if result['mean_f1'] > best_f1_macro:
            best_f1_macro = result['mean_f1']
            best_config = result
    
    # Relatório Final
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO FINAL DO GRID SEARCH")
    logger.info("=" * 70)
    
    # Tabela de resultados
    results_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
            'Acc_mean': r['mean_accuracy']
        }
        for r in all_results
    ])
    
    print("\n--- Resultados por Configuração ---")
    print(results_df.to_string(index=False))
    
    if best_config:
        print(f"\n{'='*70}")
        print("MELHOR CONFIGURAÇÃO")
        print(f"{'='*70}")
        print(f"Parâmetros: {best_config['config']}")
        print(f"F1-Score Médio: {best_config['mean_f1']:.4f} (+/- {best_config['std_f1']:.4f})")
        print(f"Accuracy Média: {best_config['mean_accuracy']:.4f}")
    
    # Salva resultados
    output_dir = Path(__file__).parent.parent / 'evaluation' / 'results' / 'toxicity_gridsearch'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocess_suffix = "_preprocess" if preprocess else "_raw"
    results_file = output_dir / f'gridsearch_{mode}{preprocess_suffix}_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': mode,
            'preprocessing': preprocess,
            'preprocessing_description': 'limpeza + lematização' if preprocess else 'texto original',
            'grid': grid,
            'results': all_results,
            'best_config': best_config,
            'test_set_size': len(X_test_final)
        }, f, indent=2, default=str)
    
    logger.info(f"\nResultados salvos em: {results_file}")
    
    print(f"\n{'='*70}")
    print(f"OBSERVAÇÃO IMPORTANTE:")
    print(f"O conjunto de TESTE FINAL ({len(X_test_final)} amostras, 20%) está reservado.")
    print(f"Use evaluate_toxicity.py para avaliar o modelo final neste conjunto.")
    print(f"{'='*70}")
    
    best_model_path = None
    if best_config:
        try:
            train_final_best_model(best_config, X_train_cv, y_train_cv, X_test_final, y_test_final)
            models_dir = Path(__file__).parent.parent / 'models' / 'trained'
            candidates = list(models_dir.glob('TOX_*'))
            if candidates:
                best_model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        except Exception as e:
            logger.error(f"Erro ao treinar e localizar modelo final: {e}")
    return all_results, best_config, best_model_path


def train_final_best_model(best_config, X_train_full, y_train_full, X_test_final, y_test_final):
    """
    Treina o modelo final usando a melhor configuração encontrada no Grid Search.
    Usa TODO o conjunto de desenvolvimento (Treino CV completo) para maximizar o aprendizado.
    """
    if not best_config:
        logger.warning("Nenhuma melhor configuração encontrada. Pulando treinamento final.")
        return

    logger.info("\n" + "=" * 70)
    logger.info("TREINANDO MODELO FINAL (BEST CONFIG)")
    logger.info("=" * 70)
    
    config = best_config['config']
    logger.info(f"Usando configuração vencedora (Idx {best_config['config_idx']}):")
    logger.info(f"{config}")
    
    # Prepara kwargs da melhor configuração
    train_kwargs = {}
    for p_name, p_val in config.items():
        if p_name in GRID_PARAM_MAPPING:
            target_arg = GRID_PARAM_MAPPING[p_name]
            train_kwargs[target_arg] = p_val
            if p_name == 'batch_size':
                train_kwargs['per_device_eval_batch_size'] = p_val

    # Aplica oversampling global (em todo o X_train_cv)
    # Importante: X_test_final continua intocado (holdout)
    logger.info("Aplicando Oversampling em todo o conjunto de treino...")
    X_train_bal, y_train_bal = apply_oversampling(
        X_train_full.tolist() if isinstance(X_train_full, np.ndarray) else X_train_full, 
        y_train_full.tolist() if isinstance(y_train_full, np.ndarray) else y_train_full
    )
    logger.info(f"Tamanho do treino final balanceado: {len(X_train_bal)}")
    
    # Instancia novo modelo
    final_model = BertimbauToxicity()
    
    # Define nome fixo para facilitar carregamento no evaluate
    # O timestamp garante que não sobrescrevemos backups, mas o evaluate busca pelo mais recente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"TOX_BEST_MODEL_{timestamp}"
    
    # Treina
    try:
        results = final_model.train_model(
            train_texts=X_train_bal,
            train_labels=y_train_bal,
            val_texts=X_test_final, # Validação apenas informativa (não usada para parada)
            val_labels=y_test_final,
            config_name='default',
            experiment_name=experiment_name,
            save_artifacts=True,
            loss_type='focal',
            focal_gamma=2.0,
            label_smoothing=0.1,
            **train_kwargs
        )
        
        output_dir = results['model_path']
        logger.info(f"MODELO FINAL TREINADO E SALVO EM: {output_dir}")
        logger.info("Agora você pode executar 'evaluate_toxicity.py' para gerar os gráficos finais.")
        
    except Exception as e:
        logger.error(f"Erro ao treinar modelo final: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Grid Search para Toxicidade')
    parser.add_argument('--mode', choices=['fast', 'full', 'optimized'], default='fast',
                        help='Modo de execução: fast (1 config), full (72 configs), '
                             'optimized (successive halving, recomendado)')
    parser.add_argument('--preprocess', action='store_true',
                        help='Aplica pré-processamento (limpeza + lematização) aos textos')
    args = parser.parse_args()
    
    all_results, best_config, best_model_path = run_grid_search(mode=args.mode, preprocess=args.preprocess)


if __name__ == "__main__":
    main()
