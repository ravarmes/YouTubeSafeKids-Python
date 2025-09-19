from .base import BaseFilter
from ..nlp.models.bertimbau_educational import BertimbauEducational
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class EducationalFilter(BaseFilter):
    """
    Filtro de conteúdo educacional usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Filtro Educacional",
            description="Identifica e avalia conteúdo educacional e adequação por idade"
        )
        
        try:
            self.model = BertimbauEducational(model_path)
            logger.info("Modelo BERTimbau educacional carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo educacional: {e}")
            self.model = None
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo para avaliar conteúdo educacional.
        
        Args:
            video_data: Dados do vídeo incluindo título, descrição e transcrição
            
        Returns:
            float: Score de 0 a 1 (0 = não educacional, 1 = muito educacional)
        """
        if not self.model:
            logger.warning("Modelo educacional não disponível")
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
            return 0.0  # Se não há texto, considera não educacional
        
        combined_text = ' '.join(text_parts)
        
        try:
            # Usa o modelo para predizer valor educacional
            result = self.model.predict_educational_value(combined_text)
            
            # Obtém o score educacional (já normalizado entre 0 e 1)
            score = self.model.get_educational_score(result)
            
            # Verifica adequação por idade se especificada
            age_group = video_data.get('target_age_group')
            if age_group:
                is_suitable = self.model.is_suitable_for_age_group(combined_text, age_group)
                if not is_suitable:
                    score *= 0.7  # Reduz score se não for adequado para a idade
            
            return score
            
        except Exception as e:
            logger.error(f"Erro ao processar conteúdo educacional: {e}")
            return 0.5  # Retorna neutro em caso de erro

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro educacional.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": self.model.get_model_info() if self.model else "Modelo não carregado",
            "options": {
                "educational_threshold": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 1.0,
                    "step": 0.1,
                    "description": "Limiar de conteúdo educacional (0.0=pouco restritivo, 1.0=muito restritivo)"
                },
                "age_group": {
                    "type": "select",
                    "options": [
                        {"value": "3-6", "label": "3-6 anos (Pré-escolar)"},
                        {"value": "7-10", "label": "7-10 anos (Fundamental I)"},
                        {"value": "11-14", "label": "11-14 anos (Fundamental II)"},
                        {"value": "15-18", "label": "15-18 anos (Ensino Médio)"}
                    ],
                    "default": "7-10",
                    "description": "Faixa etária alvo para adequação do conteúdo"
                }
            }
        }