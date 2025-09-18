"""
Classe base para modelos BERTimbau fine-tuned.

Este módulo fornece a estrutura básica para implementar modelos baseados em BERTimbau
para diferentes tarefas de classificação de texto.
"""

import os
import torch
from typing import List, Dict, Any, Union, Optional
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import logging

logger = logging.getLogger(__name__)

class BertimbauBase:
    """Classe base para modelos BERTimbau fine-tuned."""
    
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        model_path: Optional[str] = None,
        num_labels: int = 2,
        max_length: int = 128,
        device: Optional[str] = None
    ):
        """
        Inicializa o modelo BERTimbau base.
        
        Args:
            model_name: Nome do modelo base no Hugging Face (padrão: BERTimbau)
            model_path: Caminho para o modelo fine-tuned (se None, carrega o modelo base)
            num_labels: Número de classes para classificação
            max_length: Tamanho máximo da sequência de tokens
            device: Dispositivo para executar o modelo ('cuda' ou 'cpu')
        """
        self.model_name = model_name
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Define o dispositivo (GPU ou CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando modelo BERTimbau em {self.device}")
        
        # Carrega o tokenizador
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Carrega o modelo
        if model_path and os.path.exists(model_path):
            logger.info(f"Carregando modelo fine-tuned de {model_path}")
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels
            )
        else:
            logger.info(f"Carregando modelo base {model_name}")
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            
        self.model.to(self.device)
        self.model.eval()  # Modo de avaliação
        
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Realiza a predição para um texto ou lista de textos.
        
        Args:
            text: Texto ou lista de textos para classificação
            
        Returns:
            Dict contendo as predições
        """
        raise NotImplementedError("Este método deve ser implementado pela classe filha")
    
    def preprocess(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Pré-processa o texto para alimentar o modelo.
        
        Args:
            text: Texto ou lista de textos para pré-processar
            
        Returns:
            Dict com os tensores de input para o modelo
        """
        # Converte texto único para lista
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        # Tokeniza os textos
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move para o dispositivo correto (GPU/CPU)
        for key, value in encoded_inputs.items():
            encoded_inputs[key] = value.to(self.device)
            
        return encoded_inputs
    
    def save_model(self, output_dir: str):
        """
        Salva o modelo e o tokenizador em um diretório.
        
        Args:
            output_dir: Diretório para salvar o modelo
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Salvando modelo em {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir) 