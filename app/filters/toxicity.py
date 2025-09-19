from .base import BaseFilter
from ..nlp.models.bertimbau_toxicity import BertimbauToxicity
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ToxicityFilter(BaseFilter):
    """
    Filtro de toxicidade usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Filtro de Toxicidade",
            description="Detecta conteúdo tóxico, ofensivo ou inadequado"
        )
        
        try:
            self.model = BertimbauToxicity(model_path)
            logger.info("Modelo BERTimbau de toxicidade carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de toxicidade: {e}")
            self.model = None
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo para detectar toxicidade.
        
        Args:
            video_data: Dados do vídeo incluindo título, descrição e transcrição
            
        Returns:
            float: Score de 0 a 1 (0 = muito tóxico, 1 = não tóxico)
        """
        if not self.model:
            logger.warning("Modelo de toxicidade não disponível")
            return 0.5  # Retorna neutro se modelo não estiver disponível
        
        # Combina título, descrição e transcrição
        text_parts = []
        if video_data.get('title'):
            text_parts.append(video_data['title'])
        if video_data.get('description'):
            text_parts.append(video_data['description'])
        if video_data.get('transcript'):
            text_parts.append(video_data['transcript'])
        
        if not text_parts:
            return 1.0  # Se não há texto, considera não tóxico
        
        combined_text = ' '.join(text_parts)
        
        try:
            # Usa o modelo para predizer toxicidade
            result = self.model.predict_toxicity(combined_text)
            
            # Converte para score (0 = tóxico, 1 = não tóxico)
            if result['class'] == 'TOXIC':
                score = 1 - result['confidence']  # Inverte para que tóxico = baixo score
            else:
                score = result['confidence']  # Não tóxico = alto score
            
            return score
            
        except Exception as e:
            logger.error(f"Erro ao processar toxicidade: {e}")
            return 0.5  # Retorna neutro em caso de erro
        
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de toxicidade.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": self.model.get_model_info() if self.model else "Modelo não carregado",
            "options": {
                "toxicity_threshold": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                    "step": 0.1,
                    "description": "Limiar de toxicidade (0.0=muito restritivo, 1.0=pouco restritivo)"
                }
            }
        }