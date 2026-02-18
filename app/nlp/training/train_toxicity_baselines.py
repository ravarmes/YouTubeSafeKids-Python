"""
Baselines Clássicas para Detecção de Toxicidade.

Compara TF-IDF + modelos tradicionais com o BERTimbau fine-tuned.
Usa a mesma divisão de dados e metodologia (cross-validation 5 folds)
do grid search principal para garantir comparabilidade.

Modelos:
    1. SVM (LinearSVC)
    2. Logistic Regression
    3. Random Forest
    4. Naive Bayes (MultinomialNB)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Setup do path do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração
# ======================================================================

K_FOLDS = 5
RANDOM_STATE = 42
CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']

# Definição dos modelos baseline
BASELINES = {
    'SVM (LinearSVC)': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=10000
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial'
        ))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ])
}


def evaluate_with_cv(X_train_cv, y_train_cv):
    """
    Avalia todas as baselines com cross-validation estratificada (K folds).
    
    Args:
        X_train_cv: Lista de textos de treino (para CV)
        y_train_cv: Lista de labels de treino
        
    Returns:
        Dict com resultados por modelo
    """
    X = np.array(X_train_cv)
    y = np.array(y_train_cv)
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    for name, pipeline in BASELINES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Avaliando: {name}")
        logger.info(f"{'='*60}")
        
        fold_metrics = {
            'f1': [], 'accuracy': [], 'precision': [], 'recall': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Treina o pipeline (TF-IDF + Classificador)
            from sklearn.base import clone
            model = clone(pipeline)
            model.fit(X_fold_train, y_fold_train)
            
            # Predição
            y_pred = model.predict(X_fold_val)
            
            # Métricas
            f1 = f1_score(y_fold_val, y_pred, average='macro')
            acc = accuracy_score(y_fold_val, y_pred)
            prec = precision_score(y_fold_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_fold_val, y_pred, average='macro', zero_division=0)
            
            fold_metrics['f1'].append(f1)
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            
            logger.info(f"  Fold {fold}/{K_FOLDS}: F1={f1:.4f} | Acc={acc:.4f}")
        
        # Média e desvio padrão
        results[name] = {
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
            'accuracy_mean': np.mean(fold_metrics['accuracy']),
            'accuracy_std': np.std(fold_metrics['accuracy']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'fold_details': fold_metrics
        }
        
        logger.info(f"  MÉDIA: F1={results[name]['f1_mean']:.4f} (±{results[name]['f1_std']:.4f})")
    
    return results


def evaluate_on_test(X_train_cv, y_train_cv, X_test, y_test):
    """
    Treina cada baseline no conjunto de treino completo e avalia no teste.
    
    Args:
        X_train_cv: Textos de treino
        y_train_cv: Labels de treino
        X_test: Textos de teste (600 amostras reservadas)
        y_test: Labels de teste
        
    Returns:
        Dict com resultados por modelo no teste
    """
    X_train = np.array(X_train_cv)
    y_train = np.array(y_train_cv)
    X_tst = np.array(X_test)
    y_tst = np.array(y_test)
    
    test_results = {}
    
    for name, pipeline in BASELINES.items():
        logger.info(f"\n--- Teste Final: {name} ---")
        
        from sklearn.base import clone
        model = clone(pipeline)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_tst)
        
        f1 = f1_score(y_tst, y_pred, average='macro')
        acc = accuracy_score(y_tst, y_pred)
        prec = precision_score(y_tst, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_tst, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_tst, y_pred).tolist()
        report = classification_report(y_tst, y_pred, target_names=CLASS_NAMES)
        
        test_results[name] = {
            'f1': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"  F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
        logger.info(f"\n{report}")
        logger.info(f"  Matriz de Confusão:\n{np.array(cm)}")
    
    return test_results


def print_comparison_table(cv_results, test_results):
    """Imprime tabela comparativa formatada."""
    
    # --- Tabela de Cross-Validation ---
    print("\n" + "=" * 80)
    print("RESULTADOS — CROSS-VALIDATION (5 FOLDS)")
    print("=" * 80)
    print(f"{'Modelo':<25} {'F1 (macro)':<18} {'Accuracy':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 80)
    
    # Ordena por F1 decrescente
    sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    
    for name, metrics in sorted_models:
        f1_str = f"{metrics['f1_mean']:.4f} ±{metrics['f1_std']:.4f}"
        acc_str = f"{metrics['accuracy_mean']:.4f} ±{metrics['accuracy_std']:.4f}"
        prec_str = f"{metrics['precision_mean']:.4f} ±{metrics['precision_std']:.4f}"
        rec_str = f"{metrics['recall_mean']:.4f} ±{metrics['recall_std']:.4f}"
        print(f"{name:<25} {f1_str:<18} {acc_str:<18} {prec_str:<18} {rec_str:<18}")
    
    # Adiciona BERTimbau para comparação
    print("-" * 80)
    print(f"{'BERTimbau (ref.)':<25} {'0.8031 ±0.0150':<18} {'0.8036 ±---':<18} {'--- ±---':<18} {'--- ±---':<18}")
    print("=" * 80)
    
    # --- Tabela do Conjunto de Teste ---
    print("\n" + "=" * 80)
    print("RESULTADOS — CONJUNTO DE TESTE (600 amostras)")
    print("=" * 80)
    print(f"{'Modelo':<25} {'F1 (macro)':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 80)
    
    sorted_test = sorted(test_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for name, metrics in sorted_test:
        print(f"{name:<25} {metrics['f1']:<15.4f} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} {metrics['recall']:<15.4f}")
    
    print("=" * 80)


def save_results(cv_results, test_results):
    """Salva resultados em arquivo JSON."""
    output_dir = PATHS.get('evaluation_dir', os.path.join(project_root, 'results'))
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"baselines_toxicity_{timestamp}.json")
    
    # Remove fold_details para a versão salva (muito verboso)
    cv_summary = {}
    for name, metrics in cv_results.items():
        cv_summary[name] = {k: v for k, v in metrics.items() if k != 'fold_details'}
    
    output = {
        'timestamp': timestamp,
        'methodology': {
            'features': 'TF-IDF (unigrams + bigrams, max_features=10000, sublinear_tf=True)',
            'cv_folds': K_FOLDS,
            'random_state': RANDOM_STATE,
            'corpus_total': 2998,
            'train_size': 2398,
            'test_size': 600
        },
        'cross_validation_results': cv_summary,
        'test_results': test_results,
        'bertimbau_reference': {
            'cv_f1_mean': 0.8031,
            'cv_f1_std': 0.0150,
            'cv_accuracy_mean': 0.8036
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResultados salvos em: {output_file}")
    return output_file


def main():
    """Executa todas as baselines e gera relatório comparativo."""
    
    print("=" * 80)
    print("BASELINES CLÁSSICAS — DETECÇÃO DE TOXICIDADE")
    print("Comparação com BERTimbau fine-tuned")
    print("=" * 80)
    
    # 1. Carrega dados (mesma divisão do grid search)
    logger.info("Carregando dados...")
    (X_train_cv, y_train_cv), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    
    logger.info(f"Treino (CV): {len(X_train_cv)} amostras")
    logger.info(f"Teste: {len(X_test)} amostras")
    
    # 2. Cross-validation (5 folds)
    logger.info("\n>>> Iniciando Cross-Validation...")
    cv_results = evaluate_with_cv(X_train_cv, y_train_cv)
    
    # 3. Avaliação no conjunto de teste
    logger.info("\n>>> Avaliando no conjunto de teste...")
    test_results = evaluate_on_test(X_train_cv, y_train_cv, X_test, y_test)
    
    # 4. Tabela comparativa
    print_comparison_table(cv_results, test_results)
    
    # 5. Salva resultados
    output_file = save_results(cv_results, test_results)
    
    print(f"\n✓ Resultados salvos em: {output_file}")
    print("✓ Compare com BERTimbau: F1(CV) = 0.8031 ±0.0150")


if __name__ == "__main__":
    main()
