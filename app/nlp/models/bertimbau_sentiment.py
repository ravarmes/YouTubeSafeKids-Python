"""
Modelo BERTimbau para análise de sentimento em português brasileiro.

Este modelo classifica textos em sentimentos positivos, negativos ou neutros.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
import logging
from .bertimbau_base import BertimbauBase

logger = logging.getLogger(__name__)

class SentimentModel(BertimbauBase):
    """Modelo para análise de sentimento baseado no BERTimbau."""
    
    def __init__(self, model_path=None):
        """
        Inicializa o modelo de sentimento.
        
        Args:
            model_path: Caminho para o modelo fine-tuned (se None, usa o modelo base)
        """
        # Sentimento tem 3 classes: positivo, negativo, neutro
        super().__init__(
            model_name="neuralmind/bert-base-portuguese-cased",
            model_path=model_path,
            num_labels=3,
            max_length=128
        )
        
        # Mapeamento de classes
        self.id2label = {0: "negativo", 1: "neutro", 2: "positivo"}
        self.label2id = {"negativo": 0, "neutro": 1, "positivo": 2}
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Realiza a análise de sentimento para um texto ou lista de textos.
        
        Args:
            text: Texto ou lista de textos para classificar
            
        Returns:
            Dict contendo as predições de sentimento
        """
        # Pré-processamento
        inputs = self.preprocess(text)
        
        # Verifica se é um único texto ou lista
        is_single_text = isinstance(text, str)
        
        # Previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Aplica softmax para obter probabilidades
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Obtém a classe com maior probabilidade
            predictions = torch.argmax(probs, dim=-1)
        
        # Converte para numpy para processamento
        probs_np = probs.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        # Formata o resultado
        results = []
        for i, (pred, prob) in enumerate(zip(predictions_np, probs_np)):
            label = self.id2label[pred]
            confidence = float(prob[pred])
            
            # Cria dicionário com os detalhes da previsão
            result = {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    self.id2label[j]: float(p) for j, p in enumerate(prob)
                }
            }
            results.append(result)
        
        # Se for um único texto, retorna apenas o primeiro resultado
        if is_single_text:
            return results[0]
        
        return {"results": results}
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Carrega um modelo previamente treinado.
        
        Args:
            model_path: Caminho para o modelo fine-tuned
            
        Returns:
            SentimentModel: Instância do modelo carregado
        """
        return cls(model_path=model_path) 