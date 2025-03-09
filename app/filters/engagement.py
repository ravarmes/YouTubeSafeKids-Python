from typing import Dict, Any
import logging
import math
from .base import BaseFilter

logger = logging.getLogger(__name__)

class EngagementFilter(BaseFilter):
    """
    Filtro para classificar vídeos por engajamento.
    Utiliza as estatísticas da API do YouTube (visualizações, curtidas e comentários)
    para calcular um score de engajamento.
    """
    
    def __init__(self):
        super().__init__(
            name="Engajamento",
            description="Filtra por nível de engajamento do público (visualizações, curtidas e comentários)",
            default_enabled=True
        )
        # Pesos para cada métrica de engajamento
        self.view_weight = 0.4  # 40% do score
        self.like_weight = 0.4  # 40% do score
        self.comment_weight = 0.2  # 20% do score
        
        # Valores de referência para normalização
        self.view_threshold = {
            "low": 1000,      # Menos de 1000 visualizações: engajamento baixo
            "medium": 10000,  # Entre 1000 e 10000: engajamento médio
            "high": 100000    # Mais de 100000: engajamento muito alto
        }
        
        self.like_ratio_threshold = {
            "low": 0.01,    # Menos de 1% de likes por visualização: engajamento baixo
            "medium": 0.03, # Entre 1% e 3%: engajamento médio
            "high": 0.05    # Mais de 5%: engajamento alto
        }
        
        self.comment_ratio_threshold = {
            "low": 0.001,   # Menos de 0.1% de comentários por visualização: engajamento baixo
            "medium": 0.01, # Entre 0.1% e 1%: engajamento médio
            "high": 0.02    # Mais de 2%: engajamento alto
        }
        
    def _calculate_view_score(self, view_count: int) -> float:
        """
        Calcula o score com base no número de visualizações.
        Usa uma função logarítmica para normalizar valores.
        """
        if view_count <= 0:
            return 0.0
            
        # Log scale para normalizar visualizações (evita que vídeos virais monopolizem os resultados)
        log_views = math.log10(max(view_count, 1))
        log_low = math.log10(max(self.view_threshold["low"], 1))
        log_high = math.log10(max(self.view_threshold["high"], 1))
        
        # Normaliza entre 0 e 1
        normalized_score = min(1.0, max(0.0, (log_views - log_low) / (log_high - log_low)))
        self.logger.info(f"View score: {normalized_score:.4f} (from {view_count} views)")
        return normalized_score
        
    def _calculate_like_score(self, like_count: int, view_count: int) -> float:
        """
        Calcula o score com base na proporção de curtidas por visualização.
        """
        if view_count <= 0 or like_count <= 0:
            return 0.0
            
        like_ratio = like_count / view_count
        
        # Normaliza com base nos thresholds
        if like_ratio <= self.like_ratio_threshold["low"]:
            normalized_score = like_ratio / self.like_ratio_threshold["low"] * 0.33
        elif like_ratio <= self.like_ratio_threshold["medium"]:
            normalized_score = 0.33 + ((like_ratio - self.like_ratio_threshold["low"]) / 
                                     (self.like_ratio_threshold["medium"] - self.like_ratio_threshold["low"]) * 0.33)
        else:
            normalized_score = 0.66 + min(0.34, ((like_ratio - self.like_ratio_threshold["medium"]) / 
                                              (self.like_ratio_threshold["high"] - self.like_ratio_threshold["medium"]) * 0.34))
        
        self.logger.info(f"Like score: {normalized_score:.4f} (ratio: {like_ratio:.4f}, {like_count} likes / {view_count} views)")
        return normalized_score
        
    def _calculate_comment_score(self, comment_count: int, view_count: int) -> float:
        """
        Calcula o score com base na proporção de comentários por visualização.
        """
        if view_count <= 0 or comment_count <= 0:
            return 0.0
            
        comment_ratio = comment_count / view_count
        
        # Normaliza com base nos thresholds
        if comment_ratio <= self.comment_ratio_threshold["low"]:
            normalized_score = comment_ratio / self.comment_ratio_threshold["low"] * 0.33
        elif comment_ratio <= self.comment_ratio_threshold["medium"]:
            normalized_score = 0.33 + ((comment_ratio - self.comment_ratio_threshold["low"]) / 
                                     (self.comment_ratio_threshold["medium"] - self.comment_ratio_threshold["low"]) * 0.33)
        else:
            normalized_score = 0.66 + min(0.34, ((comment_ratio - self.comment_ratio_threshold["medium"]) / 
                                              (self.comment_ratio_threshold["high"] - self.comment_ratio_threshold["medium"]) * 0.34))
        
        self.logger.info(f"Comment score: {normalized_score:.4f} (ratio: {comment_ratio:.4f}, {comment_count} comments / {view_count} views)")
        return normalized_score
            
    def process(self, video: Dict[str, Any]) -> float:
        """
        Processa o engajamento do vídeo e retorna um score entre 0 e 1.
        Usa as estatísticas do vídeo (viewCount, likeCount, commentCount).
        """
        try:
            if not self.validate_video(video):
                self.logger.warning(f"Invalid video data for EngagementFilter")
                return 0
                
            self.logger.info(f"Processing engagement for video: {video['title']}")
            
            # Obtém estatísticas
            view_count = int(video.get('view_count', 0))
            like_count = int(video.get('like_count', 0))
            comment_count = int(video.get('comment_count', 0))
            
            self.logger.info(f"Video stats: {view_count} views, {like_count} likes, {comment_count} comments")
            
            # Calcula scores individuais
            view_score = self._calculate_view_score(view_count)
            like_score = self._calculate_like_score(like_count, view_count)
            comment_score = self._calculate_comment_score(comment_count, view_count)
            
            # Calcula score final com pesos
            final_score = (
                view_score * self.view_weight +
                like_score * self.like_weight +
                comment_score * self.comment_weight
            )
            
            self.logger.info(f"Final engagement score: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error processing engagement: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0
            
    def get_filter_info(self):
        """
        Retorna informações sobre o filtro.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": "engagement",
            "default_value": 100,  # Máximo engajamento por padrão
            "options": {
                "engagement_levels": [
                    {"value": "all", "label": "Todos os níveis"},
                    {"value": "low", "label": "Baixo engajamento"},
                    {"value": "medium", "label": "Engajamento médio"},
                    {"value": "high", "label": "Alto engajamento"}
                ]
            }
        } 