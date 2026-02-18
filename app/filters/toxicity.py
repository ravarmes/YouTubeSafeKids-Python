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
            name="Toxicidade",
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
            
            # Classes: 0=Não Tóxico, 1=Levemente Tóxico, 2=Moderadamente Tóxico, 3=Altamente Tóxico
            # Score: 1 = não tóxico (bom), 0 = altamente tóxico (ruim)
            predicted_class = result['predicted_class']
            
            if 'probabilities' in result:
                # Calcula score usando probabilidades ponderadas
                # Pesos: quanto mais tóxico, menor o score
                class_weights = {
                    'Não Tóxico': 1.0,
                    'Levemente Tóxico': 0.6,
                    'Moderadamente Tóxico': 0.3,
                    'Altamente Tóxico': 0.0
                }
                score = sum(
                    result['probabilities'].get(label, 0) * weight 
                    for label, weight in class_weights.items()
                )
            else:
                # Fallback: calcula baseado apenas na classe predita
                score = 1.0 - (predicted_class / 3.0)
            
            return min(max(score, 0.0), 1.0)
            
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