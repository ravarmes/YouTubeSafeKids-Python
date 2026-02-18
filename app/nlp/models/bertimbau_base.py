"""Classe base para modelos BERTimbau do projeto YouTube Safe Kids.

Esta classe fornece uma interface comum para todos os modelos de classificação
baseados no BERTimbau, facilitando a implementação e manutenção.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from datetime import datetime
from ..config import MODEL_CONFIG, get_task_config, PATHS
from ..utils.data_utils import DataProcessor
from ..utils.evaluation_utils import ModelEvaluator

logger = logging.getLogger(__name__)

class BertimbauBase:
    """
    Classe base para modelos BERTimbau.
    
    Esta classe fornece funcionalidades comuns para todos os modelos
    de classificação do projeto, incluindo carregamento, predição,
    pré-processamento e salvamento.
    """
    
    def __init__(
        self,
        task_name: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa o modelo base.
        
        Args:
            task_name: Nome da tarefa (AS, TOX, LI, TE)
            model_path: Caminho para modelo pré-treinado (opcional)
            device: Dispositivo para execução (cuda/cpu)
            config: Configurações personalizadas (opcional)
        """
        self.task_name = task_name
        
        # Carrega configuração da tarefa
        self.task_config = get_task_config(task_name)
        self.num_labels = self.task_config['num_labels']
        self.class_names = self.task_config['classes']
        
        # Configurações do modelo
        self.model_config = MODEL_CONFIG.copy()
        if config:
            self.model_config.update(config)
        
        self.max_length = self.model_config['max_length']
        
        # Configura dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Inicializa modelo e tokenizador
        self.model = None
        self.tokenizer = None
        self.training_metadata = {}
        
        # Carrega modelo se caminho fornecido
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_base_model()
        
        logger.info(f"Modelo {task_name} ({self.task_config['name']}) inicializado no dispositivo {self.device}")
    
    def _initialize_base_model(self):
        """Inicializa modelo base BERTimbau."""
        model_name = self.model_config['base_model']
        cache_dir = self.model_config.get('cache_dir')
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels, cache_dir=cache_dir)
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels)
        
        self.model.to(self.device)
        
    def predict(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Realiza a predição para um texto.
        
        Args:
            text: Texto para classificação
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Dict contendo as predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Use load_model() primeiro.")
        
        self.model.eval()
        
        # Pré-processa o texto
        inputs = self.preprocess(text)
        
        # Realiza predição
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Calcula probabilidades
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            result = {
                'predicted_class': predicted_class,
                'predicted_label': self.class_names[predicted_class],
                'confidence': probabilities[0][predicted_class].item()
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    self.class_names[i]: prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
        
        return result
    
    def predict_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Realiza predições em lote.
        
        Args:
            texts: Lista de textos para classificação
            batch_size: Tamanho do lote
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Lista com predições para cada texto
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Use load_model() primeiro.")
        
        self.model.eval()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Pré-processa o lote
            inputs = self.preprocess_batch(batch_texts)
            
            # Realiza predições
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Calcula probabilidades
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                
                # Processa resultados do lote
                for j, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
                    result = {
                        'predicted_class': pred_class.item(),
                        'predicted_label': self.class_names[pred_class.item()],
                        'confidence': probs[pred_class].item()
                    }
                    
                    if return_probabilities:
                        result['probabilities'] = {
                            self.class_names[k]: prob.item()
                            for k, prob in enumerate(probs)
                        }
                    
                    results.append(result)
        
        return results
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Pré-processa um texto para alimentar o modelo.
        
        Args:
            text: Texto para pré-processar
            
        Returns:
            Dict com os tensores de input para o modelo
        """
        # Tokeniza o texto
        encoded_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move para o dispositivo correto
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        
        return encoded_inputs
    
    def preprocess_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Pré-processa uma lista de textos para alimentar o modelo.
        
        Args:
            texts: Lista de textos para pré-processar
            
        Returns:
            Dict com os tensores de input para o modelo
        """
        # Tokeniza os textos
        encoded_inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move para o dispositivo correto
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        
        return encoded_inputs
    
    def load_model(self, model_path: str):
        """
        Carrega um modelo pré-treinado.
        
        Args:
            model_path: Caminho para o modelo
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        logger.info(f"Carregando modelo de {model_path}")
        
        # Carrega tokenizador e modelo
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        
        self.model.to(self.device)
        
        # Carrega metadados se existirem
        metadata_path = os.path.join(model_path, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.training_metadata = json.load(f)
        
        logger.info(f"Modelo {self.task_name} carregado com sucesso")
    
    def save_model(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Salva o modelo e o tokenizador em um diretório.
        
        Args:
            output_dir: Diretório para salvar o modelo
            metadata: Metadados adicionais para salvar
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Modelo não foi inicializado")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Salvando modelo em {output_dir}")
        
        # Salva modelo e tokenizador
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Salva metadados
        model_metadata = {
            'task_name': self.task_name,
            'task_config': self.task_config,
            'model_config': self.model_config,
            'save_date': datetime.now().isoformat(),
            'num_labels': self.num_labels,
            'class_names': self.class_names,
            'max_length': self.max_length
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        if self.training_metadata:
            model_metadata['training_metadata'] = self.training_metadata
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo {self.task_name} salvo com sucesso em {output_dir}")
    
    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int],
        batch_size: int = 32,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Avalia o modelo em um conjunto de teste.
        
        Args:
            test_texts: Textos de teste
            test_labels: Labels verdadeiros
            batch_size: Tamanho do lote para predições
            save_results: Se deve salvar os resultados
            output_dir: Diretório para salvar resultados
            
        Returns:
            Dict com métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado")
        
        # Cria avaliador
        if output_dir is None:
            output_dir = os.path.join(PATHS['evaluation_dir'], self.task_name)
        
        evaluator = ModelEvaluator(
            task_name=self.task_name,
            class_names=self.class_names,
            output_dir=output_dir
        )
        
        # Realiza avaliação
        results = evaluator.evaluate_model(
            model=self.model,
            tokenizer=self.tokenizer,
            test_texts=test_texts,
            test_labels=test_labels,
            max_length=self.max_length,
            batch_size=batch_size
        )
        
        # Salva relatório se solicitado
        if save_results:
            model_path = getattr(self, 'model_path', 'modelo_atual')
            evaluator.generate_report(results, model_path)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.
        
        Returns:
            Dict com informações do modelo
        """
        info = {
            'task_name': self.task_name,
            'task_description': self.task_config['name'],
            'num_labels': self.num_labels,
            'class_names': self.class_names,
            'max_length': self.max_length,
            'device': str(self.device),
            'base_model': self.model_config['base_model'],
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None
        }
        
        if self.training_metadata:
            info['training_metadata'] = self.training_metadata
        
        return info
    
    def __str__(self) -> str:
        """Representação em string do modelo."""
        return f"BertimbauBase(task={self.task_name}, labels={self.num_labels}, device={self.device})"
    
    def __repr__(self) -> str:
        """Representação detalhada do modelo."""
        return self.__str__()
