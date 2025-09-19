from .base import BaseFilter
from ..nlp.models.bertimbau_sentiment import BertimbauSentiment
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SentimentFilter(BaseFilter):
    """
    Filtro para análise de sentimento usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Sentimento",
            description="Filtra por sentimento usando modelo BERTimbau especializado",
            default_enabled=True
        )
        try:
            self.model = BertimbauSentiment(model_path=model_path)
            logger.info("Modelo de sentimentos carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            self.model = None
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        O score é baseado na análise de sentimento do modelo especializado.
        """
        if self.model is None:
            logger.warning("Modelo não disponível, retornando score neutro")
            return 0.5
            
        text = f"{video_data.get('title', '')} {video_data.get('description', '')} {video_data.get('transcript', '')}"
        if not text.strip():
            return 0.5  # Neutro quando não há texto
            
        try:
            # Usa o modelo especializado para análise
            result = self.model.predict_sentiment(text, return_probabilities=True)
            
            # Converte a classe predita em score (0-1)
            # 0=Negativo, 1=Neutro, 2=Positivo
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Score baseado na classe e confiança
            if predicted_class == 0:  # Negativo
                score = 0.2 * confidence  # Score baixo para sentimento negativo
            elif predicted_class == 1:  # Neutro
                score = 0.5  # Score neutro
            else:  # Positivo
                score = 0.8 + (0.2 * confidence)  # Score alto para sentimento positivo
                
            return min(max(score, 0.0), 1.0)  # Garante que está entre 0 e 1
            
        except Exception as e:
            logger.error(f"Erro ao processar sentimento: {e}")
            return 0.5  # Retorna neutro em caso de erro
    
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de sentimento.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": self.model.get_model_info() if self.model else "Modelo não carregado",
            "options": {
                "sentiment_preference": {
                    "type": "slider",
                    "min": 0,
                    "max": 100,
                    "default": 50,
                    "description": "Preferência de sentimento (0=negativo, 50=neutro, 100=positivo)"
                }
            }
        }