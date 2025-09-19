from .base import BaseFilter
from ..nlp.models.bertimbau_language import BertimbauLanguage
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LanguageFilter(BaseFilter):
    """
    Filtro de linguagem imprópria usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Filtro de Linguagem",
            description="Detecta linguagem imprópria, palavrões e conteúdo inadequado"
        )
        
        try:
            self.model = BertimbauLanguage(model_path)
            logger.info("Modelo BERTimbau de linguagem carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de linguagem: {e}")
            self.model = None
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo para detectar linguagem imprópria.
        
        Args:
            video_data: Dados do vídeo incluindo título, descrição e transcrição
            
        Returns:
            float: Score de 0 a 1 (0 = linguagem muito imprópria, 1 = linguagem apropriada)
        """
        if not self.model:
            logger.warning("Modelo de linguagem não disponível")
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
            return 1.0  # Se não há texto, considera sem linguagem imprópria
        
        combined_text = ' '.join(text_parts)
        
        try:
            # Usa o modelo para predizer adequação da linguagem
            result = self.model.predict_language_appropriateness(combined_text)
            
            # Converte para score (0 = severa, 1 = nenhuma)
            if result['class'] == 'INAPPROPRIATE':
                score = 1 - result['confidence']  # Inverte para que impróprio = baixo score
            else:
                score = result['confidence']  # Nenhuma = alto score
            
            return score
            
        except Exception as e:
            logger.error(f"Erro ao processar linguagem: {e}")
            return 0.5  # Retorna neutro em caso de erro
        
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de linguagem.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": self.model.get_model_info() if self.model else "Modelo não carregado",
            "options": {
                "language_threshold": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                    "step": 0.1,
                    "description": "Limiar de linguagem apropriada (0.0=muito restritivo, 1.0=pouco restritivo)"
                }
            }
        }