"""
Script de treinamento com Cross-Validation (K=5) para Análise de Sentimentos.
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

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
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
    """
    df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Separa classes
    df_neg = df[df['label'] == 0]
    df_neu = df[df['label'] == 1]
    df_pos = df[df['label'] == 2]
    
    # Define alvo (classe majoritária)
    max_count = max(len(df_neg), len(df_neu), len(df_pos))
    
    # Upsample
    df_neg_up = resample(df_neg, replace=True, n_samples=max_count, random_state=42)
    df_neu_up = resample(df_neu, replace=True, n_samples=max_count, random_state=42)
    df_pos_up = resample(df_pos, replace=True, n_samples=max_count, random_state=42)
    
    df_balanced = pd.concat([df_neg_up, df_neu_up, df_pos_up])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()

def main():
    logger.info("=" * 60)
    logger.info("INICIANDO CROSS-VALIDATION (K=5) - METODOLOGIA 80/20")
    logger.info("=" * 60)
    
    # 1. Obter dados (80% Treino para CV, 20% Teste Reservado)
    # Aqui garantimos a divisão pedida pelo professor
    (X_train_cv_list, y_train_cv_list), (X_test_final, y_test_final) = get_data_for_cv_and_test(test_size=0.20)
    
    # Converter para numpy para indexação fácil no KFold
    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    
    fold_metrics = []
    K_FOLDS = 5
    
    # 2. Loop de Treinamento (Acontece DENTRO dos 80%)
    # get_kfold_split vai dividir os 80% em pedaços de treino/validação internos
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
        model = BertimbauSentiment()
        
        # Treinar
        results = model.train_model(
            train_texts=X_train_bal,
            train_labels=y_train_bal,
            val_texts=X_fold_val,
            val_labels=y_fold_val,
            config_name='AS_best',
            experiment_name=f'sentiment_cv_fold_{curr_fold}'
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