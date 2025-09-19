"""
Avaliador de modelos para o projeto YouTube Safe Kids.

Este módulo fornece uma classe para avaliar modelos de forma padronizada,
gerando relatórios detalhados e visualizações.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Classe para avaliação de modelos."""
    
    def __init__(
        self,
        task_name: str,
        class_names: List[str],
        output_dir: str = "./evaluation_results"
    ):
        """
        Inicializa o avaliador.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            class_names: Nomes das classes
            output_dir: Diretório para salvar resultados
        """
        self.task_name = task_name
        self.class_names = class_names
        self.output_dir = output_dir
        
        # Cria diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        test_texts: List[str],
        test_labels: List[int],
        max_length: int = 128,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            model: Modelo treinado
            tokenizer: Tokenizador
            test_texts: Textos de teste
            test_labels: Labels verdadeiros
            max_length: Comprimento máximo das sequências
            batch_size: Tamanho do batch
            
        Returns:
            Dict com métricas e predições
        """
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        probabilities = []
        
        # Processa em batches
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i+batch_size]
            
            # Tokeniza
            inputs = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predição
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calcula métricas
        metrics = self._calculate_metrics(test_labels, predictions, probabilities)
        
        # Adiciona predições aos resultados
        metrics['predictions'] = predictions.tolist()
        metrics['probabilities'] = probabilities.tolist()
        metrics['true_labels'] = test_labels
        
        logger.info(f"Avaliação concluída para {self.task_name}")
        logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        true_labels: List[int],
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula métricas de avaliação.
        
        Args:
            true_labels: Labels verdadeiros
            predictions: Predições do modelo
            probabilities: Probabilidades das predições
            
        Returns:
            Dict com métricas
        """
        # Métricas básicas
        accuracy = accuracy_score(true_labels, predictions)
        
        # Precision, Recall, F1
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Por classe
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        # Matriz de confusão
        cm = confusion_matrix(true_labels, predictions)
        
        # AUC-ROC (apenas para classificação binária ou multiclasse)
        auc_scores = {}
        if len(self.class_names) == 2:
            # Binária
            auc_scores['binary'] = roc_auc_score(true_labels, probabilities[:, 1])
        elif len(self.class_names) > 2:
            # Multiclasse - one-vs-rest
            try:
                auc_scores['macro'] = roc_auc_score(
                    true_labels, probabilities, multi_class='ovr', average='macro'
                )
                auc_scores['weighted'] = roc_auc_score(
                    true_labels, probabilities, multi_class='ovr', average='weighted'
                )
            except ValueError:
                logger.warning("Não foi possível calcular AUC-ROC")
        
        # Relatório de classificação
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'auc_scores': auc_scores,
            'classification_report': class_report
        }
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        model_path: str,
        save_plots: bool = True
    ) -> str:
        """
        Gera relatório detalhado da avaliação.
        
        Args:
            metrics: Métricas calculadas
            model_path: Caminho do modelo avaliado
            save_plots: Se deve salvar gráficos
            
        Returns:
            Caminho do arquivo de relatório
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(
            self.output_dir, 
            f"evaluation_report_{self.task_name}_{timestamp}.txt"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE AVALIAÇÃO - {self.task_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Data da avaliação: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Modelo avaliado: {model_path}\n")
            f.write(f"Tarefa: {self.task_name}\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n\n")
            
            # Métricas gerais
            f.write("MÉTRICAS GERAIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Acurácia: {metrics['accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Precisão (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n\n")
            
            # AUC-ROC
            if metrics['auc_scores']:
                f.write("AUC-ROC\n")
                f.write("-" * 10 + "\n")
                for key, value in metrics['auc_scores'].items():
                    f.write(f"AUC-ROC ({key}): {value:.4f}\n")
                f.write("\n")
            
            # Métricas por classe
            f.write("MÉTRICAS POR CLASSE\n")
            f.write("-" * 25 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precisão: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"  Suporte: {metrics['support_per_class'][i]}\n\n")
            
            # Matriz de confusão
            f.write("MATRIZ DE CONFUSÃO\n")
            f.write("-" * 20 + "\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write("Predito →\n")
            f.write("Real ↓\n")
            
            # Cabeçalho
            f.write("        ")
            for class_name in self.class_names:
                f.write(f"{class_name:>8}")
            f.write("\n")
            
            # Linhas da matriz
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name:>8}")
                for j in range(len(self.class_names)):
                    f.write(f"{cm[i][j]:>8}")
                f.write("\n")
        
        # Salva métricas em JSON
        json_file = os.path.join(
            self.output_dir,
            f"evaluation_metrics_{self.task_name}_{timestamp}.json"
        )
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Gera gráficos se solicitado
        if save_plots:
            self._save_plots(metrics, timestamp)
        
        logger.info(f"Relatório salvo em {report_file}")
        return report_file
    
    def _save_plots(self, metrics: Dict[str, Any], timestamp: str):
        """
        Salva gráficos da avaliação.
        
        Args:
            metrics: Métricas calculadas
            timestamp: Timestamp para nomes dos arquivos
        """
        try:
            # Matriz de confusão
            plt.figure(figsize=(8, 6))
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title(f'Matriz de Confusão - {self.task_name.upper()}')
            plt.ylabel('Real')
            plt.xlabel('Predito')
            plt.tight_layout()
            
            plot_file = os.path.join(
                self.output_dir,
                f"confusion_matrix_{self.task_name}_{timestamp}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Métricas por classe
            plt.figure(figsize=(10, 6))
            x = np.arange(len(self.class_names))
            width = 0.25
            
            plt.bar(x - width, metrics['precision_per_class'], width, label='Precisão')
            plt.bar(x, metrics['recall_per_class'], width, label='Recall')
            plt.bar(x + width, metrics['f1_per_class'], width, label='F1-Score')
            
            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title(f'Métricas por Classe - {self.task_name.upper()}')
            plt.xticks(x, self.class_names)
            plt.legend()
            plt.ylim(0, 1)
            plt.tight_layout()
            
            plot_file = os.path.join(
                self.output_dir,
                f"metrics_by_class_{self.task_name}_{timestamp}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráficos salvos em {self.output_dir}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar gráficos: {e}")
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compara resultados de múltiplos modelos.
        
        Args:
            model_results: Dict com resultados de cada modelo
            save_comparison: Se deve salvar comparação
            
        Returns:
            Dict com comparação
        """
        comparison = {
            'models': list(model_results.keys()),
            'metrics_comparison': {}
        }
        
        # Métricas para comparar
        metrics_to_compare = [
            'accuracy', 'f1_macro', 'f1_weighted',
            'precision_macro', 'recall_macro'
        ]
        
        for metric in metrics_to_compare:
            comparison['metrics_comparison'][metric] = {
                model_name: results[metric]
                for model_name, results in model_results.items()
            }
        
        # Encontra melhor modelo para cada métrica
        comparison['best_models'] = {}
        for metric in metrics_to_compare:
            best_model = max(
                comparison['metrics_comparison'][metric].items(),
                key=lambda x: x[1]
            )
            comparison['best_models'][metric] = {
                'model': best_model[0],
                'value': best_model[1]
            }
        
        if save_comparison:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_file = os.path.join(
                self.output_dir,
                f"model_comparison_{self.task_name}_{timestamp}.json"
            )
            
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comparação salva em {comparison_file}")
        
        return comparison