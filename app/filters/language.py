from .base import BaseFilter
from ..nlp.models.bertimbau_language import BertimbauLanguage
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LanguageFilter(BaseFilter):
    """
    Linguagem Imprópria usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Linguagem Imprópria",
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
            
            # Classes: 0=Nenhuma, 1=Leve, 2=Severa
            # Score: 1 = linguagem apropriada (bom), 0 = linguagem severa (ruim)
            predicted_class = result['predicted_class']
            
            if 'probabilities' in result:
                # Calcula score usando probabilidades ponderadas
                # Pesos: quanto mais impróprio, menor o score
                class_weights = {
                    'Nenhuma': 1.0,
                    'Leve': 0.5,
                    'Severa': 0.0
                }
                score = sum(
                    result['probabilities'].get(label, 0) * weight 
                    for label, weight in class_weights.items()
                )
            else:
                # Fallback: calcula baseado apenas na classe predita
                score = 1.0 - (predicted_class / 2.0)
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Erro ao processar linguagem: {e}")
            return 0.5  # Retorna neutro em caso de erro
        
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de linguagem imprópria.
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