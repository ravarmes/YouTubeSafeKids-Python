from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .base import BaseFilter
from .duration import DurationFilter
from .age_rating import AgeRatingFilter
from .educational import EducationalFilter
from .toxicity import ToxicityFilter
from .language import LanguageFilter
from .diversity import DiversityFilter
from .interactivity import InteractivityFilter
from .engagement import EngagementFilter
from .sentiment import SentimentFilter
from .sensitive import SensitiveFilter
import logging

__all__ = [
    'BaseFilter',
    'DurationFilter',
    'AgeRatingFilter',
    'EducationalFilter',
    'ToxicityFilter',
    'LanguageFilter',
    'DiversityFilter',
    'InteractivityFilter',
    'EngagementFilter',
    'SentimentFilter',
    'SensitiveFilter'
]

logger = logging.getLogger(__name__)

class FilterManager:
    """Manages all filters in the application."""
    
    def __init__(self):
        self.filters: Dict[str, BaseFilter] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing FilterManager")
    
    def register_filter(self, name: str, filter_instance: BaseFilter) -> None:
        """Register a new filter."""
        self.filters[name] = filter_instance
        self.logger.info(f"Registered filter: {name}")
    
    def get_filter(self, name: str) -> BaseFilter:
        """Get a filter by name."""
        return self.filters.get(name)
    
    def get_all_filters(self) -> Dict[str, BaseFilter]:
        """Get all registered filters."""
        return self.filters
    
    def get_enabled_filters(self, filter_weights: Dict[str, Any]) -> Dict[str, BaseFilter]:
        """
        Get only the filters that are enabled and have weights.
        
        Args:
            filter_weights: Dictionary of filter weights from the frontend
            
        Returns:
            Dict[str, BaseFilter]: Enabled filters with their weights
        """
        enabled_filters = {}
        
        # Log todos os filtros disponíveis
        self.logger.info(f"=== Verificando filtros habilitados ===")
        self.logger.info(f"Total de filtros disponíveis: {len(self.filters)}")
        self.logger.info(f"Filtros disponíveis: {', '.join(self.filters.keys())}")
        
        # Log de filter_weights recebido
        if not filter_weights:
            self.logger.warning("Nenhum filtro habilitado recebido!")
            return {}
            
        self.logger.info(f"Filtros recebidos do frontend: {', '.join(filter_weights.keys())}")
        
        # Verifica cada filtro disponível
        for name, filter_instance in self.filters.items():
            # Só inclui filtros que estão no dicionário filter_weights
            if name in filter_weights:
                enabled_filters[name] = filter_instance
                self.logger.info(f"✓ Filtro '{name}' está habilitado com configuração: {filter_weights[name]}")
            else:
                self.logger.info(f"✗ Filtro '{name}' não está habilitado (ignorado)")
        
        # Verifica se há filtros que foram enviados mas não existem
        for name in filter_weights:
            if name not in self.filters:
                self.logger.warning(f"⚠ Filtro '{name}' foi enviado mas não existe no sistema")
        
        # Retorna apenas os filtros habilitados
        self.logger.info(f"Total de filtros habilitados: {len(enabled_filters)}")
        if enabled_filters:
            self.logger.info(f"Filtros habilitados: {', '.join(enabled_filters.keys())}")
        else:
            self.logger.warning("Nenhum filtro será aplicado!")
            
        return enabled_filters
    
    def process_video(self, video: Dict[str, Any], filter_weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a video through enabled filters only.
        
        Args:
            video: Dictionary containing video information
            filter_weights: Dictionary of filter weights from the frontend
            
        Returns:
            Dict[str, Any]: Video with filter scores and final score
        """
        self.logger.info(f"\n=== Processando vídeo ===")
        self.logger.info(f"Título: {video.get('title', 'Desconhecido')}")
        self.logger.info(f"ID: {video.get('id', 'Desconhecido')}")
        
        if 'duration_seconds' in video:
            minutes = video['duration_seconds'] / 60
            self.logger.info(f"Duração: {minutes:.2f} minutos ({video['duration_seconds']} segundos)")
        elif 'duration' in video:
            self.logger.info(f"Duração ISO: {video['duration']}")
        
        # Verificar se o vídeo tem os campos necessários
        if not video or 'id' not in video or 'title' not in video:
            self.logger.warning("⚠ Vídeo inválido, faltando campos obrigatórios")
            video['filter_scores'] = {}
            video['final_score'] = 0
            return video
        
        # Get only enabled filters with their weights - apenas os filtros presentes em filter_weights
        enabled_filters = self.get_enabled_filters(filter_weights)
        
        # Se não houver filtros habilitados, retorna o vídeo sem processamento
        if not enabled_filters:
            self.logger.warning("⚠ Nenhum filtro habilitado para processar este vídeo")
            video['filter_scores'] = {}
            video['final_score'] = 0
            return video
        
        # Process each filter
        filter_scores = {}
        total_weight = 0
        weighted_sum = 0
        
        self.logger.info(f"Aplicando {len(enabled_filters)} filtros ao vídeo:")
        
        for name, filter_instance in enabled_filters.items():
            self.logger.info(f"\n▶ Processando filtro: {name}")
            
            try:
                # Get filter weight from frontend
                weight_info = filter_weights[name]
                
                # Debug do weight_info
                self.logger.info(f"Configuração do filtro: {weight_info}")
                
                # Handle weight_info in different formats
                weight = 1.0  # Peso padrão
                
                if isinstance(weight_info, dict):
                    # Extrai o peso (se disponível) ou usa o padrão 1.0
                    weight = float(weight_info.get('weight', 1.0))
                    
                    # Set filter type if provided (para filtros como duração, faixa etária, etc)
                    if 'type' in weight_info:
                        filter_type = weight_info['type']
                        self.logger.info(f"Definindo tipo do filtro para '{filter_type}'")
                        
                        # Caso especial para o filtro de duração que usa o nome em lowercase
                        if name == "Duração":
                            key = "duração_type"
                            self.logger.info(f"Caso especial para Duração: definindo {key}={filter_type}")
                            video[key] = filter_type
                        else:
                            # Para outros filtros
                            key = f"{name.lower()}_type"
                            video[key] = filter_type
                elif isinstance(weight_info, (int, float)):
                    # Se for um número, usa diretamente como peso
                    weight = float(weight_info)
                else:
                    self.logger.warning(f"⚠ Formato desconhecido para {name}: {weight_info}, usando peso padrão 1.0")
                
                # Calculate filter score
                self.logger.info(f"Calculando score para o filtro {name}")
                score = filter_instance.process(video)
                self.logger.info(f"Score do filtro {name}: {score:.4f}")
                
                filter_scores[name] = score
                
                # Add to weighted sum
                weighted_score = score * weight
                weighted_sum += weighted_score
                total_weight += weight
                
                self.logger.info(f"Filtro {name}: Score {score:.4f} × Peso {weight:.2f} = {weighted_score:.4f}")
            except Exception as e:
                self.logger.error(f"⚠ Erro ao processar filtro {name}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Atribui score zero em caso de erro
                filter_scores[name] = 0
        
        # Calculate final score
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        self.logger.info(f"\nCálculo do score final: {weighted_sum:.4f} ÷ {total_weight:.2f} = {final_score:.4f}")
        
        result = {
            **video,
            'filter_scores': filter_scores,
            'final_score': final_score
        }
        
        # Log com todos os scores para facilitar debug
        self.logger.info("\n=== Resumo dos scores ===")
        for name, score in filter_scores.items():
            self.logger.info(f"- {name}: {score:.4f}")
        self.logger.info(f"Score final: {final_score:.4f}")
        
        return result
    
    def get_filter_info(self) -> List[Dict[str, Any]]:
        """Get information about all filters."""
        return [
            filter_instance.get_filter_info()
            for filter_instance in self.filters.values()
        ]

# Create global filter manager instance
filter_manager = FilterManager()

def get_all_filters() -> dict[str, BaseFilter]:
    """
    Retorna um dicionário com todos os filtros disponíveis.
    """
    return filter_manager.filters 