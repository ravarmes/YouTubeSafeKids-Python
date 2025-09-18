"""
Modelo BERTimbau para detecção de toxicidade em conteúdo em português brasileiro.

Este modelo identifica conteúdo tóxico, ofensivo ou inadequado para crianças.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
import logging
from .bertimbau_base import BertimbauBase

logger = logging.getLogger(__name__)

class ToxicityModel(BertimbauBase):
    """Modelo para detecção de toxicidade baseado no BERTimbau."""
    
    def __init__(self, model_path=None):
        """
        Inicializa o modelo de detecção de toxicidade.
        
        Args:
            model_path: Caminho para o modelo fine-tuned (se None, usa o modelo base)
        """
        # Toxicidade pode ser um problema de classificação binária ou multi-classe
        # Aqui usamos classificação binária: tóxico vs. não-tóxico
        super().__init__(
            model_name="neuralmind/bert-base-portuguese-cased",
            model_path=model_path,
            num_labels=2,
            max_length=256  # Sequências mais longas para capturar contexto
        )
        
        # Mapeamento de classes
        self.id2label = {0: "não-tóxico", 1: "tóxico"}
        self.label2id = {"não-tóxico": 0, "tóxico": 1}
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Realiza a detecção de toxicidade para um texto ou lista de textos.
        
        Args:
            text: Texto ou lista de textos para classificar
            
        Returns:
            Dict contendo as predições de toxicidade
        """
        # Pré-processamento
        inputs = self.preprocess(text)
        
        # Verifica se é um único texto ou lista
        is_single_text = isinstance(text, str)
        
        # Previsão
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Aplica sigmoid para obter probabilidades (para classificação binária)
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
            toxicity_score = float(prob[1])  # probabilidade da classe tóxica
            
            # Cria dicionário com os detalhes da previsão
            result = {
                "label": label,
                "is_toxic": bool(pred == 1),
                "toxicity_score": toxicity_score,
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
            ToxicityModel: Instância do modelo carregado
        """
        return cls(model_path=model_path) 