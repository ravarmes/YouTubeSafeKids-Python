"""
Modelo BERTimbau para classificação de valor educacional em conteúdo em português brasileiro.

Este modelo avalia o valor educacional de textos, classificando-os em diferentes níveis ou categorias.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
import logging
from .bertimbau_base import BertimbauBase

logger = logging.getLogger(__name__)

class EducationalModel(BertimbauBase):
    """Modelo para classificação de valor educacional baseado no BERTimbau."""
    
    def __init__(self, model_path=None):
        """
        Inicializa o modelo de classificação educacional.
        
        Args:
            model_path: Caminho para o modelo fine-tuned (se None, usa o modelo base)
        """
        # Classificação em 4 níveis educacionais
        super().__init__(
            model_name="neuralmind/bert-base-portuguese-cased",
            model_path=model_path,
            num_labels=4,
            max_length=256  # Sequências mais longas para capturar contexto
        )
        
        # Mapeamento de classes
        self.id2label = {
            0: "não-educacional", 
            1: "baixo_valor_educacional", 
            2: "médio_valor_educacional", 
            3: "alto_valor_educacional"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Realiza a classificação educacional para um texto ou lista de textos.
        
        Args:
            text: Texto ou lista de textos para classificar
            
        Returns:
            Dict contendo as predições de valor educacional
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
            edu_score = float(
                # Pontuação normalizada entre 0 e 1 com base na classe
                # Quanto maior o índice da classe, maior o valor educacional
                pred / (len(self.id2label) - 1)
            )
            
            # Cria dicionário com os detalhes da previsão
            result = {
                "label": label,
                "educational_value": edu_score,
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
            EducationalModel: Instância do modelo carregado
        """
        return cls(model_path=model_path) 