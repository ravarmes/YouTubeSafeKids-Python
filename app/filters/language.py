from transformers import pipeline
from .base import BaseFilter
from typing import Dict, Any

class LanguageFilter(BaseFilter):
    """
    Filtro para classificar vídeos por linguagem imprópria.
    """
    
    def __init__(self):
        super().__init__(
            name="Linguagem Imprópria",
            description="Filtra por linguagem imprópria",
            default_enabled=True
        )
        self.classifier = pipeline("text-classification", model="neuralmind/bert-base-portuguese-cased")
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        0 = linguagem mais imprópria, 1 = linguagem mais apropriada
        """
        text = f"{video_data.get('title', '')} {video_data.get('description', '')} {video_data.get('transcript', '')}"
        if not text.strip():
            return 1.0
            
        # Divide o texto em chunks de no máximo 512 tokens
        chunks = self._split_text(text)
        
        # Processa cada chunk
        scores = []
        for chunk in chunks:
            try:
                result = self.classifier(chunk)
                # Inverte o score para que 1 seja o mais apropriado
                score = 1 - result[0]['score'] if result[0]['label'] == 'IMPROPRIA' else result[0]['score']
                scores.append(score)
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
                continue
                
        if not scores:
            return 1.0
            
        # Retorna a média dos scores
        return sum(scores) / len(scores)
        
    def _split_text(self, text: str, max_length: int = 512) -> list:
        """
        Divide o texto em chunks menores para processamento.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": "language",
            "default_value": 0,
            "options": {
                "language_levels": [
                    {"value": "all", "label": "Todos os níveis"},
                    {"value": "appropriate", "label": "Apropriada"},
                    {"value": "moderate", "label": "Moderada"},
                    {"value": "inappropriate", "label": "Inapropriada"}
                ]
            }
        } 