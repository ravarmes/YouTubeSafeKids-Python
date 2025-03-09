"""
Modelo BERTimbau para detecção de linguagem imprópria em português brasileiro.

Este modelo identifica linguagem inadequada para crianças, incluindo palavrões, 
gírias ofensivas e termos inapropriados.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
import logging
from .bertimbau_base import BertimbauBase

logger = logging.getLogger(__name__)

class LanguageModel(BertimbauBase):
    """Modelo para detecção de linguagem imprópria baseado no BERTimbau."""
    
    def __init__(self, model_path=None):
        """
        Inicializa o modelo de detecção de linguagem imprópria.
        
        Args:
            model_path: Caminho para o modelo fine-tuned (se None, usa o modelo base)
        """
        # Classificação em 3 níveis de adequação de linguagem
        super().__init__(
            model_name="neuralmind/bert-base-portuguese-cased",
            model_path=model_path,
            num_labels=3,
            max_length=256  # Sequências mais longas para capturar contexto
        )
        
        # Mapeamento de classes
        self.id2label = {
            0: "linguagem_apropriada", 
            1: "linguagem_questionável", 
            2: "linguagem_imprópria"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Realiza a detecção de linguagem imprópria para um texto ou lista de textos.
        
        Args:
            text: Texto ou lista de textos para classificar
            
        Returns:
            Dict contendo as predições sobre adequação da linguagem
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
            
            # Pontuação de impropriedade (quanto maior, mais imprópria)
            impropriety_score = float(pred / (len(self.id2label) - 1))
            
            # Cria dicionário com os detalhes da previsão
            result = {
                "label": label,
                "impropriety_score": impropriety_score,
                "is_appropriate": bool(pred == 0),
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
            LanguageModel: Instância do modelo carregado
        """
        return cls(model_path=model_path) 