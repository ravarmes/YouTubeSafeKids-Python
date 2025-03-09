from typing import Dict, Any, List, Optional
import re
from transformers import BertTokenizer
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseFilter(ABC):
    """
    Classe base para todos os filtros.
    Define a interface comum que todos os filtros devem implementar.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        default_enabled: bool = True,
        weight: float = 1.0
    ):
        logger.info(f"{self.__class__.__name__} :: def __init__")
        self.name = name
        self.description = description
        self.enabled = default_enabled
        self.weight = weight
        self.logger = logging.getLogger(f"filters.{name.lower()}")
        self.logger.info(f"Initializing {name} filter")
        self.tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        
    @abstractmethod
    def process(self, video: Dict[str, Any]) -> float:
        """
        Process a video and return a score between 0 and 1.
        
        Args:
            video: Dictionary containing video information
            
        Returns:
            float: Score between 0 and 1
        """
        pass
        
    @abstractmethod
    def get_filter_info(self) -> Dict[str, Any]:
        """
        Get information about the filter.
        
        Returns:
            Dict[str, Any]: Filter information including name, description, and options
        """
        pass
        
    def is_enabled(self) -> bool:
        logger.info(f"{self.__class__.__name__} :: def is_enabled")
        """Check if the filter is enabled."""
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        logger.info(f"{self.__class__.__name__} :: def set_enabled")
        """Enable or disable the filter."""
        self.enabled = enabled
        self.logger.info(f"{self.name} filter {'enabled' if enabled else 'disabled'}")
    
    def get_weight(self) -> float:
        logger.info(f"{self.__class__.__name__} :: def get_weight")
        """Get the filter's weight."""
        return self.weight
    
    def set_weight(self, weight: float) -> None:
        logger.info(f"{self.__class__.__name__} :: def set_weight")
        """Set the filter's weight."""
        self.weight = weight
        self.logger.info(f"{self.name} filter weight set to {weight}")
    
    def validate_video(self, video: Dict[str, Any]) -> bool:
        logger.info(f"{self.__class__.__name__} :: def validate_video")
        """
        Validate if a video has all required fields.
        
        Args:
            video: Dictionary containing video information
            
        Returns:
            bool: True if video is valid, False otherwise
        """
        required_fields = ['id', 'title', 'duration']
        for field in required_fields:
            if field not in video:
                self.logger.warning(f"Video missing required field: {field}")
                return False
        return True
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Divide o texto em chunks menores para processamento usando o tokenizer do BERT.
        Garante que cada chunk tenha no máximo max_length tokens.
        """
        # Remove espaços extras e quebras de linha
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokeniza o texto
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            
            if current_length >= max_length:
                # Converte os tokens de volta para texto e adiciona ao chunk
                chunk_text = self.tokenizer.convert_tokens_to_string(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
                
        # Adiciona o último chunk se houver tokens restantes
        if current_chunk:
            chunk_text = self.tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
            
        return chunks 