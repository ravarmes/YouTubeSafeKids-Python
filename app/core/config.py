from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from typing import Dict, List, ClassVar
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "YouTube Safe Kids"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
    
    # Configuração de busca
    MAX_SEARCH_RESULTS: int = 4
    
    # Configuração de transcrição
    ENABLE_VIDEO_TRANSCRIPTION: bool = True

    # Configurações dos filtros
    FILTER_WEIGHTS: ClassVar[Dict[str, float]] = {
        "duration": 1.0,
        "engagement": 1.0,
        "age_rating": 1.0,
        "interactivity": 1.0,
        "language": 1.0,
        "toxicity": 1.0,
        "sentiment": 1.0,
        "educational": 1.0,
        "diversity": 1.0,
        "sensitive": 1.0
    }

    # Filtros que requerem transcrição de vídeo
    NLP_FILTER_NAMES: ClassVar[List[str]] = [
        "Sentimento", "Toxicidade", "Tópicos Educacionais", "Linguagem Imprópria"
    ]

    class Config:
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()