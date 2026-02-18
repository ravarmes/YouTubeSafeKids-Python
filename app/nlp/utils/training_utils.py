"""
Utilitários para treinamento de modelos.

Este módulo fornece classes e funções para facilitar o treinamento
dos modelos BERTimbau de forma padronizada.
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

class TrainingHelper:
    """Classe auxiliar para treinamento de modelos."""
    
    def __init__(
        self,
        task_name: str,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        output_base_dir: str = "./models"
    ):
        """
        Inicializa o helper de treinamento.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            model_name: Nome do modelo base
            output_base_dir: Diretório base para salvar modelos
        """
        self.task_name = task_name
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        
        # Cria diretório de saída se não existir
        os.makedirs(output_base_dir, exist_ok=True)
        
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: List[str],
        test_labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepara os datasets para treinamento.
        
        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            val_texts: Textos de validação
            val_labels: Labels de validação
            test_texts: Textos de teste
            test_labels: Labels de teste
            tokenizer: Tokenizador BERT
            max_length: Comprimento máximo das sequências
            
        Returns:
            Tuple com datasets tokenizados
        """
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Cria datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'labels': test_labels
        })
        
        # Tokeniza
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Datasets preparados para {self.task_name}")
        logger.info(f"Treino: {len(train_dataset)} amostras")
        logger.info(f"Validação: {len(val_dataset)} amostras")
        logger.info(f"Teste: {len(test_dataset)} amostras")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_f1",
        greater_is_better: bool = True
    ) -> TrainingArguments:
        """
        Cria argumentos de treinamento padronizados.
        
        Args:
            output_dir: Diretório de saída
            num_train_epochs: Número de épocas
            per_device_train_batch_size: Batch size para treino
            per_device_eval_batch_size: Batch size para avaliação
            learning_rate: Taxa de aprendizado
            warmup_steps: Passos de warmup
            weight_decay: Decaimento de peso
            logging_steps: Frequência de logging
            eval_steps: Frequência de avaliação
            save_steps: Frequência de salvamento
            evaluation_strategy: Estratégia de avaliação
            save_strategy: Estratégia de salvamento
            load_best_model_at_end: Carregar melhor modelo ao final
            metric_for_best_model: Métrica para seleção do melhor modelo
            greater_is_better: Se maior valor da métrica é melhor
            
        Returns:
            TrainingArguments configurado
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=None,  # Desabilita wandb/tensorboard por padrão
            seed=42
        )
    
    def compute_metrics(self, eval_pred):
        """
        Computa métricas durante o treinamento.
        
        Args:
            eval_pred: Predições e labels verdadeiros
            
        Returns:
            Dict com métricas
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_labels: int,
        training_args: TrainingArguments,
        model_path: Optional[str] = None
    ) -> Tuple[BertForSequenceClassification, Trainer]:
        """
        Treina o modelo.
        
        Args:
            train_dataset: Dataset de treino
            val_dataset: Dataset de validação
            num_labels: Número de classes
            training_args: Argumentos de treinamento
            model_path: Caminho para modelo pré-treinado (opcional)
            
        Returns:
            Tuple com modelo treinado e trainer
        """
        # Carrega o modelo
        if model_path and os.path.exists(model_path):
            model = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
        
        # Cria o trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info(f"Iniciando treinamento para {self.task_name}")
        
        # Treina o modelo
        trainer.train()
        
        logger.info(f"Treinamento concluído para {self.task_name}")
        
        return model, trainer
    
    def save_model_with_metadata(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        output_dir: str,
        training_args: TrainingArguments,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Salva o modelo com metadados.
        
        Args:
            model: Modelo treinado
            tokenizer: Tokenizador
            output_dir: Diretório de saída
            training_args: Argumentos de treinamento
            metrics: Métricas finais
            additional_info: Informações adicionais
        """
        # Salva modelo e tokenizador
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Cria metadados
        metadata = {
            'task': self.task_name,
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'training_args': {
                'num_train_epochs': training_args.num_train_epochs,
                'per_device_train_batch_size': training_args.per_device_train_batch_size,
                'learning_rate': training_args.learning_rate,
                'warmup_steps': training_args.warmup_steps,
                'weight_decay': training_args.weight_decay
            },
            'final_metrics': metrics
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        # Salva metadados
        with open(os.path.join(output_dir, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo salvo em {output_dir} com metadados")
    
    def get_output_dir(self, experiment_name: Optional[str] = None) -> str:
        """
        Gera diretório de saída para o modelo.
        
        Args:
            experiment_name: Nome do experimento (opcional)
            
        Returns:
            Caminho do diretório de saída
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            dir_name = f"{self.task_name}_{experiment_name}_{timestamp}"
        else:
            dir_name = f"{self.task_name}_{timestamp}"
        
        return os.path.join(self.output_base_dir, dir_name)


def create_training_config(
    task: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Cria configuração padrão para treinamento.
    
    Args:
        task: Nome da tarefa
        epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        max_length: Comprimento máximo das sequências
        
    Returns:
        Dict com configuração
    """
    return {
        'task': task,
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'num_train_epochs': epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'eval_steps': 500,
        'save_steps': 500,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_f1',
        'greater_is_better': True
    }