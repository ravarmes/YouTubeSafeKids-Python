from .base import BaseFilter
from ..nlp.models.bertimbau_sentiment import BertimbauSentiment
from typing import Dict, Any
import logging
import os
from pathlib import Path

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
        
        # --- LÓGICA AUTOMÁTICA PARA ENCONTRAR O MODELO TREINADO ---
        if model_path is None:
            try:
                # Caminho base onde os modelos são salvos
                current_dir = Path(__file__).parent
                models_dir = current_dir.parent / 'nlp' / 'models' / 'trained'
                
                # Procura pastas que começam com 'AS_' (Análise de Sentimentos)
                if models_dir.exists():
                    model_paths = list(models_dir.glob('AS_*'))
                    if model_paths:
                        # Pega o mais recente (pela data de modificação)
                        latest_model = max(model_paths, key=lambda p: p.stat().st_mtime)
                        model_path = str(latest_model)
                        logger.info(f"Modelo de sentimento encontrado automaticamente: {model_path}")
                    else:
                        logger.warning("Nenhum modelo treinado 'AS_*' encontrado na pasta trained.")
                else:
                    logger.warning(f"Diretório de modelos não encontrado: {models_dir}")
            except Exception as e:
                logger.error(f"Erro ao tentar localizar modelo automaticamente: {e}")

        # ----------------------------------------------------------

        try:
            self.model = BertimbauSentiment(model_path=model_path)
            logger.info(f"Modelo de sentimentos carregado com sucesso de: {model_path}")
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
            
        # Junta título, descrição e transcrição para análise completa
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
            
            # Lógica de pontuação refinada para segurança infantil
            if predicted_class == 0:  # Negativo
                # Penaliza fortemente conteúdo negativo
                # Quanto maior a confiança de que é negativo, menor a nota (mais perigoso)
                score = 0.1 + (0.2 * (1 - confidence)) 
            elif predicted_class == 1:  # Neutro
                score = 0.5  # Score base
            else:  # Positivo
                # Bonifica conteúdo positivo
                score = 0.7 + (0.3 * confidence)
                
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
            "model_info": "BERTimbau Fine-tuned (AS)" if self.model else "Modelo não carregado",
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