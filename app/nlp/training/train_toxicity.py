"""
Script de treinamento com Cross-Validation (K=5) para Detecção de Toxicidade.
Metodologia:
1. Recebe o conjunto de TREINO (80% do total).
2. Realiza Cross-Validation interno dividindo esse treino em 5 folds.
3. Aplica Oversampling APENAS na parte de treino de cada fold.
"""

import logging
import sys
import os
import numpy as np
import pandas as pd
from sklearn.utils import resample

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_oversampling(train_texts, train_labels):
    """
    Balanceia classes minoritárias replicando dados (Oversampling).
    Isso é feito APENAS no fold de treino para evitar vazamento de dados.
    
    Classes: 0=Nenhuma, 1=Leve, 2=Severa
    """
    df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Separa classes
    df_none = df[df['label'] == 0]  # Nenhuma
    df_mild = df[df['label'] == 1]  # Leve
    df_severe = df[df['label'] == 2]  # Severa
    
    # Define alvo (classe majoritária)
    max_count = max(len(df_none), len(df_mild), len(df_severe))
    
    # Upsample
    df_none_up = resample(df_none, replace=True, n_samples=max_count, random_state=42)
    df_mild_up = resample(df_mild, replace=True, n_samples=max_count, random_state=42)
    df_severe_up = resample(df_severe, replace=True, n_samples=max_count, random_state=42)
    
    df_balanced = pd.concat([df_none_up, df_mild_up, df_severe_up])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()


def main():
    logger.info("=" * 60)
    logger.info("INICIANDO CROSS-VALIDATION (K=5) - TOXICIDADE")
    logger.info("Metodologia: 80% Treino (CV) / 20% Teste (Reservado)")
    logger.info("=" * 60)
    
    # 1. Obter dados (80% Treino para CV, 20% Teste Reservado)
    (X_train_cv_list, y_train_cv_list), (X_test_final, y_test_final) = get_data_for_cv_and_test(test_size=0.20)
    
    # Converter para numpy para indexação fácil no KFold
    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    
    fold_metrics = []
    K_FOLDS = 5
    
    # 2. Loop de Treinamento (Acontece DENTRO dos 80%)
    for fold, (train_idx, val_idx) in enumerate(get_kfold_split(X_train_cv, y_train_cv, n_splits=K_FOLDS)):
        curr_fold = fold + 1
        logger.info(f"\n>>> [FOLD {curr_fold}/{K_FOLDS}] Iniciando Rodada...")
        
        # Separação interna do fold (Treino vs Validação)
        X_fold_train, X_fold_val = X_train_cv[train_idx], X_train_cv[val_idx]
        y_fold_train, y_fold_val = y_train_cv[train_idx], y_train_cv[val_idx]
        
        # Oversampling (APENAS no Treino do fold atual)
        X_train_bal, y_train_bal = apply_oversampling(X_fold_train.tolist(), y_fold_train.tolist())
        
        # Converter validação para lista
        X_fold_val = X_fold_val.tolist()
        y_fold_val = y_fold_val.tolist()
        
        logger.info(f"  Status: Treino Balanceado={len(X_train_bal)} | Validação Interna={len(X_fold_val)}")
        
        # Instanciar novo modelo limpo
        model = BertimbauToxicity()
        
        # Treinar
        results = model.train_model(
            train_texts=X_train_bal,
            train_labels=y_train_bal,
            val_texts=X_fold_val,
            val_labels=y_fold_val,
            config_name='TOX_best',
            experiment_name=f'toxicity_cv_fold_{curr_fold}',
            save_artifacts=False,
            loss_type='focal',
            focal_gamma=2.0,
            label_smoothing=0.1
        )
        
        # Coletar métricas
        metrics = results['final_metrics']
        metrics['fold'] = curr_fold
        fold_metrics.append(metrics)
        
        acc = metrics.get('eval_accuracy', metrics.get('accuracy', 0))
        logger.info(f"  [FOLD {curr_fold}] Concluído. Acurácia na Validação: {acc:.4f}")

    # 3. Relatório Final
    logger.info("=" * 60)
    logger.info("RESUMO FINAL DO CROSS-VALIDATION (MÉDIA DOS FOLDS)")
    logger.info("=" * 60)
    
    df_results = pd.DataFrame(fold_metrics)
    
    # Médias e Desvios
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    means = df_results[numeric_cols].mean()
    stds = df_results[numeric_cols].std()
    
    print("\n--- Tabela de Resultados por Fold ---")
    cols_show = ['fold'] + [c for c in ['eval_accuracy', 'accuracy', 'eval_loss', 'loss'] if c in df_results.columns]
    print(df_results[cols_show].to_string(index=False))
    
    print("\n--- Média Geral (+/- Desvio Padrão) ---")
    for col in numeric_cols:
        if col != 'fold':
            print(f"{col}: {means[col]:.4f} (+/- {stds[col]:.4f})")
            
    print("\n" + "=" * 60)
    print(f"OBSERVAÇÃO IMPORTANTE:")
    print(f"O conjunto de TESTE FINAL ({len(X_test_final)} amostras, 20% do total) está separado.")
    print("Ele NÃO foi usado em nenhum momento acima.")
    print("Se desejar, você pode carregar o melhor modelo salvo e avaliá-lo nesse conjunto.")
    print("=" * 60)


if __name__ == "__main__":
    main()
