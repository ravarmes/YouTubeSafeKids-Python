from transformers import pipeline
from .base import BaseFilter
from typing import Dict, Any

class ToxicityFilter(BaseFilter):
    """
    Filtro para detectar conteúdo tóxico usando BERTimbau.
    """
    
    def __init__(self):
        super().__init__(
            name="Toxicidade",
            description="Filtra por conteúdo tóxico",
            default_enabled=True
        )
        self.classifier = pipeline("text-classification", model="neuralmind/bert-base-portuguese-cased")
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        0 = mais tóxico, 1 = menos tóxico
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
                # Inverte o score para que 1 seja o menos tóxico
                score = 1 - result[0]['score'] if result[0]['label'] == 'TOXICO' else result[0]['score']
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
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": "toxicity",
            "default_value": 0,
            "options": {
                "toxicity_levels": [
                    {"value": "all", "label": "Todos os níveis"},
                    {"value": "low", "label": "Baixa toxicidade"},
                    {"value": "medium", "label": "Toxicidade média"},
                    {"value": "high", "label": "Alta toxicidade"}
                ]
            }
        } 