"""
Script para fine-tuning de modelos BERTimbau para tarefas específicas.

Este script fornece as funções básicas para realizar o fine-tuning do modelo BERTimbau
para tarefas de classificação de texto, como análise de sentimento, detecção de toxicidade,
classificação educacional e detecção de linguagem imprópria.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str, text_column: str, label_column: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Carrega um dataset a partir de um arquivo CSV ou Excel.
    
    Args:
        file_path: Caminho para o arquivo de dados
        text_column: Nome da coluna que contém o texto
        label_column: Nome da coluna que contém os rótulos
        
    Returns:
        Tuple com datasets de treino, validação e teste
    """
    logger.info(f"Carregando dataset de {file_path}")
    
    # Detecta a extensão do arquivo
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Carrega o dataset
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {file_ext}")
    
    logger.info(f"Dataset carregado com {len(df)} exemplos")
    
    # Divide em treino, validação e teste (70%, 15%, 15%)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logger.info(f"Divisão do dataset - Treino: {len(train_df)}, Validação: {len(val_df)}, Teste: {len(test_df)}")
    
    # Converte para o formato datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset

def preprocess_data(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer: BertTokenizer,
    text_column: str,
    label_column: str,
    max_length: int = 128
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Pré-processa os datasets para treino.
    
    Args:
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        test_dataset: Dataset de teste
        tokenizer: Tokenizador BERT
        text_column: Nome da coluna de texto
        label_column: Nome da coluna de rótulos
        max_length: Tamanho máximo da sequência
        
    Returns:
        Datasets processados
    """
    logger.info(f"Pré-processando datasets com max_length={max_length}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokeniza os datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Configura formato para treinamento
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', label_column])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', label_column])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', label_column])
    
    return train_dataset, val_dataset, test_dataset

def compute_metrics(pred):
    """
    Calcula métricas de avaliação.
    
    Args:
        pred: Predições do modelo
        
    Returns:
        Dict com as métricas calculadas
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calcula métricas de avaliação
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_labels: int,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 5e-5
) -> BertForSequenceClassification:
    """
    Treina um modelo BERTimbau para uma tarefa específica.
    
    Args:
        model_name: Nome do modelo base no Hugging Face
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        num_labels: Número de classes
        output_dir: Diretório para salvar o modelo treinado
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        
    Returns:
        Modelo treinado
    """
    logger.info(f"Iniciando treinamento do modelo {model_name} com {num_labels} classes")
    
    # Carrega o modelo base
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Configura os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=learning_rate,
    )
    
    # Configura o trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Treina o modelo
    logger.info("Iniciando treinamento...")
    trainer.train()
    
    # Salva o modelo final
    logger.info(f"Salvando modelo treinado em {output_dir}")
    trainer.save_model(output_dir)
    
    return model

def evaluate_model(model, test_dataset: Dataset) -> Dict[str, float]:
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo treinado
        test_dataset: Dataset de teste
        
    Returns:
        Dict com as métricas de avaliação
    """
    logger.info("Avaliando modelo no conjunto de teste")
    
    # Configura o trainer para avaliação
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    
    # Avalia o modelo
    results = trainer.evaluate(test_dataset)
    
    logger.info(f"Resultados da avaliação: {results}")
    
    return results

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Treinamento de modelo BERTimbau')
    
    parser.add_argument('--task', type=str, required=True, 
                        choices=['sentiment', 'toxicity', 'educational', 'language'],
                        help='Tarefa para a qual o modelo será treinado')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Caminho para o arquivo de dados')
    
    parser.add_argument('--text_column', type=str, default='text',
                        help='Nome da coluna que contém o texto')
    
    parser.add_argument('--label_column', type=str, default='label',
                        help='Nome da coluna que contém os rótulos')
    
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Diretório para salvar o modelo treinado')
    
    parser.add_argument('--model_name', type=str, 
                        default='neuralmind/bert-base-portuguese-cased',
                        help='Nome do modelo base no Hugging Face')
    
    parser.add_argument('--num_labels', type=int, default=0,
                        help='Número de classes (se 0, usa o padrão da tarefa)')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Número de épocas de treinamento')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamanho do batch')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Taxa de aprendizado')
    
    parser.add_argument('--max_length', type=int, default=128,
                        help='Tamanho máximo da sequência')
    
    args = parser.parse_args()
    
    # Define o número de classes com base na tarefa se não especificado
    if args.num_labels == 0:
        task_labels = {
            'sentiment': 3,      # positivo, negativo, neutro
            'toxicity': 2,       # tóxico, não-tóxico
            'educational': 4,    # não-educacional, baixo, médio, alto
            'language': 3        # apropriada, questionável, imprópria
        }
        args.num_labels = task_labels[args.task]
    
    # Define o diretório de saída específico para a tarefa
    model_output_dir = os.path.join(args.output_dir, f"bertimbau_{args.task}")
    
    # Carrega o tokenizador
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Carrega e pré-processa os dados
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.data_path, args.text_column, args.label_column
    )
    
    train_dataset, val_dataset, test_dataset = preprocess_data(
        train_dataset, val_dataset, test_dataset,
        tokenizer, args.text_column, args.label_column,
        args.max_length
    )
    
    # Treina o modelo
    model = train_model(
        args.model_name,
        train_dataset,
        val_dataset,
        args.num_labels,
        model_output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # Avalia o modelo
    results = evaluate_model(model, test_dataset)
    
    # Salva os resultados
    results_file = os.path.join(model_output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Treinamento concluído. Modelo salvo em {model_output_dir}")

if __name__ == "__main__":
    main() 