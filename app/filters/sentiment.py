from transformers import pipeline
from .base import BaseFilter
from typing import Dict, Any

class SentimentFilter(BaseFilter):
    """
    Filtro para análise de sentimento usando BERTimbau.
    """
    
    def __init__(self):
        super().__init__(
            name="Sentimento",
            description="Filtra por sentimento",
            default_enabled=True
        )
        self.classifier = pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        O score é baseado na confiança da classificação e no tipo de sentimento desejado.
        """
        text = f"{video_data.get('title', '')} {video_data.get('description', '')} {video_data.get('transcript', '')}"
        if not text.strip():
            return 0.5  # Neutro quando não há texto
            
        # Divide o texto em chunks de no máximo 512 tokens
        chunks = self._split_text(text)
        
        # Processa cada chunk
        sentiments = []
        confidences = []
        
        for chunk in chunks:
            try:
                result = self.classifier(chunk)[0]
                sentiments.append(result['label'])
                confidences.append(result['score'])
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
                continue
                
        if not sentiments:
            return 0.5  # Neutro quando não há resultados
            
        # Conta a frequência de cada sentimento
        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")
        total = len(sentiments)
        
        # Calcula a proporção de cada sentimento
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        neutral_ratio = 1 - (positive_ratio + negative_ratio)
        
        # Calcula a confiança média
        avg_confidence = sum(confidences) / len(confidences)
        
        # Obtem o valor do filtro de sentimento (0-100)
        # Onde 0 = prefere conteúdo triste, 100 = prefere conteúdo alegre, 50 = neutro
        sentiment_value = video_data.get('Sentimento', 50)
        
        # Normaliza para um valor entre 0 e 1
        normalized_value = sentiment_value / 100
        
        # Calcula o score baseado na preferência de sentimento
        # - Se o valor for baixo (triste), multiplica o ratio negativo por um peso maior
        # - Se o valor for alto (alegre), multiplica o ratio positivo por um peso maior
        # - Se o valor for meio (neutro), valoriza conteúdo neutro
        if normalized_value < 0.33:  # Preferência por conteúdo triste
            score = negative_ratio * avg_confidence
        elif normalized_value > 0.66:  # Preferência por conteúdo alegre
            score = positive_ratio * avg_confidence
        else:  # Preferência por conteúdo neutro
            score = neutral_ratio * avg_confidence
            
        return score
            
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
            "weight": self.weight,
            "type": "sentiment",
            "default_value": 100
        } 