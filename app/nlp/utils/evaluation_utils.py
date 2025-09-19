"""
Utilitários para avaliação de modelos.

Este módulo fornece classes e funções para avaliar o desempenho
dos modelos treinados de forma padronizada.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Classe para avaliação padronizada de modelos."""
    
    def __init__(self, task_name: str, label_names: Optional[Dict[int, str]] = None):
        """
        Inicializa o avaliador.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            label_names: Mapeamento de índices para nomes das classes
        """
        self.task_name = task_name
        self.label_names = label_names or {}
        
    def evaluate(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_proba: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Avalia as predições do modelo.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            y_proba: Probabilidades das classes (opcional)
            
        Returns:
            Dict com métricas de avaliação
        """
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Métricas por classe
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Relatório de classificação
        class_report = classification_report(
            y_true, y_pred, 
            target_names=[self.label_names.get(i, f'Classe_{i}') for i in range(len(set(y_true)))],
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'support': support
        }
        
        logger.info(f"Avaliação concluída para {self.task_name}")
        logger.info(f"Acurácia: {accuracy:.4f}")
        logger.info(f"F1-Score (weighted): {f1:.4f}")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """
        Imprime os resultados de forma formatada.
        
        Args:
            results: Resultados da avaliação
        """
        print(f"\n=== Resultados da Avaliação - {self.task_name} ===")
        print(f"Acurácia: {results['accuracy']:.4f}")
        print(f"Precisão (weighted): {results['precision_weighted']:.4f}")
        print(f"Recall (weighted): {results['recall_weighted']:.4f}")
        print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
        
        print("\n--- Métricas por Classe ---")
        for i, (p, r, f) in enumerate(zip(
            results['precision_per_class'],
            results['recall_per_class'], 
            results['f1_per_class']
        )):
            class_name = self.label_names.get(i, f'Classe_{i}')
            print(f"{class_name}: Precisão={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    
    def plot_confusion_matrix(
        self, 
        results: Dict[str, Any], 
        save_path: Optional[str] = None
    ):
        """
        Plota a matriz de confusão.
        
        Args:
            results: Resultados da avaliação
            save_path: Caminho para salvar o gráfico (opcional)
        """
        try:
            cm = np.array(results['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=[self.label_names.get(i, f'Classe_{i}') for i in range(cm.shape[1])],
                yticklabels=[self.label_names.get(i, f'Classe_{i}') for i in range(cm.shape[0])]
            )
            plt.title(f'Matriz de Confusão - {self.task_name}')
            plt.ylabel('Classe Verdadeira')
            plt.xlabel('Classe Predita')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Matriz de confusão salva em {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn não disponível. Pulando visualização.")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Salva os resultados em um arquivo JSON.
        
        Args:
            results: Resultados da avaliação
            output_path: Caminho para salvar o arquivo
        """
        import json
        
        # Adiciona metadados
        results_with_meta = {
            'task': self.task_name,
            'label_names': self.label_names,
            'metrics': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados salvos em {output_path}")


def compare_models(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compara múltiplas avaliações de modelos.
    
    Args:
        evaluations: Dict com nome_modelo -> resultados_avaliacao
        
    Returns:
        DataFrame com comparação das métricas
    """
    comparison_data = []
    
    for model_name, results in evaluations.items():
        comparison_data.append({
            'Modelo': model_name,
            'Acurácia': results['accuracy'],
            'Precisão': results['precision_weighted'],
            'Recall': results['recall_weighted'],
            'F1-Score': results['f1_weighted']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df


def generate_evaluation_report(
    task_name: str,
    results: Dict[str, Any],
    model_info: Dict[str, Any],
    output_path: str
):
    """
    Gera um relatório completo de avaliação.
    
    Args:
        task_name: Nome da tarefa
        results: Resultados da avaliação
        model_info: Informações do modelo (hiperparâmetros, etc.)
        output_path: Caminho para salvar o relatório
    """
    report = f"""
# Relatório de Avaliação - {task_name}

## Informações do Modelo
- **Tarefa**: {task_name}
- **Modelo Base**: {model_info.get('model_name', 'N/A')}
- **Épocas de Treinamento**: {model_info.get('epochs', 'N/A')}
- **Batch Size**: {model_info.get('batch_size', 'N/A')}
- **Learning Rate**: {model_info.get('learning_rate', 'N/A')}

## Métricas Gerais
- **Acurácia**: {results['accuracy']:.4f}
- **Precisão (weighted)**: {results['precision_weighted']:.4f}
- **Recall (weighted)**: {results['recall_weighted']:.4f}
- **F1-Score (weighted)**: {results['f1_weighted']:.4f}

## Métricas por Classe
"""
    
    for i, (p, r, f) in enumerate(zip(
        results['precision_per_class'],
        results['recall_per_class'],
        results['f1_per_class']
    )):
        report += f"- **Classe {i}**: Precisão={p:.4f}, Recall={r:.4f}, F1={f:.4f}\n"
    
    report += f"""
## Matriz de Confusão
```
{np.array(results['confusion_matrix'])}
```

## Relatório de Classificação Detalhado
```
{results['classification_report']}
```
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Relatório de avaliação salvo em {output_path}")