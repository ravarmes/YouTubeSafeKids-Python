"""
Módulo de pré-processamento de texto para NLP.

Etapas implementadas conforme TCC:
- Limpeza de texto (remoção de pontuação, aspas, caracteres especiais, lowercase)
- Lematização/Stemming (usando NLTK RSLPStemmer para português)

Nota: Tokenização e Vetorização são feitas internamente pelo BERTimbau.
      Remoção de stopwords NÃO é recomendada para modelos BERT.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Flag para controlar se NLTK está disponível
NLTK_AVAILABLE = False
stemmer = None

def _load_nltk():
    """Carrega o stemmer NLTK (lazy loading)."""
    global NLTK_AVAILABLE, stemmer
    
    if stemmer is not None:
        return True
    
    try:
        import nltk
        from nltk.stem import RSLPStemmer
        
        # Verifica se o recurso RSLP está disponível
        try:
            stemmer = RSLPStemmer()
            NLTK_AVAILABLE = True
            logger.info("NLTK RSLPStemmer carregado para português")
            return True
        except LookupError:
            # Baixa o recurso se não estiver disponível
            logger.info("Baixando recurso RSLP do NLTK...")
            nltk.download('rslp', quiet=True)
            stemmer = RSLPStemmer()
            NLTK_AVAILABLE = True
            logger.info("NLTK RSLPStemmer carregado para português")
            return True
    except ImportError:
        logger.warning("NLTK não instalado. Instale com: pip install nltk")
        return False


def clean_text(text: str) -> str:
    """
    Limpeza básica de texto.
    
    Etapas:
    1. Remove URLs
    2. Remove menções (@usuario)
    3. Remove hashtags
    4. Remove pontuação e caracteres especiais
    5. Remove números
    6. Converte para minúsculas
    7. Remove espaços múltiplos
    
    Args:
        text: Texto original
        
    Returns:
        Texto limpo
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove menções
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove emojis e caracteres especiais unicode (mantém acentos)
    text = re.sub(r'[^\w\s\u00C0-\u00FF]', ' ', text)
    
    # Remove números
    text = re.sub(r'\d+', '', text)
    
    # Converte para minúsculas
    text = text.lower()
    
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def lemmatize_text(text: str) -> str:
    """
    Aplica stemming/lematização ao texto usando NLTK RSLPStemmer.
    
    O RSLPStemmer é um stemmer para português que reduz palavras
    aos seus radicais, similar à lematização.
    
    Exemplo: "correndo" -> "corr", "gatos" -> "gat"
    
    Args:
        text: Texto para processar
        
    Returns:
        Texto com palavras reduzidas aos radicais
    """
    if not _load_nltk():
        logger.warning("Lematização não disponível, retornando texto original")
        return text
    
    if not isinstance(text, str) or not text.strip():
        return ""
    
    words = text.split()
    stems = [stemmer.stem(word) for word in words]
    return ' '.join(stems)


def preprocess_text(text: str, apply_cleaning: bool = True, apply_lemmatization: bool = True) -> str:
    """
    Pipeline completo de pré-processamento.
    
    Args:
        text: Texto original
        apply_cleaning: Se True, aplica limpeza de texto
        apply_lemmatization: Se True, aplica lematização/stemming
        
    Returns:
        Texto pré-processado
    """
    result = text
    
    if apply_cleaning:
        result = clean_text(result)
    
    if apply_lemmatization:
        result = lemmatize_text(result)
    
    return result


def preprocess_batch(texts: List[str], apply_cleaning: bool = True, 
                     apply_lemmatization: bool = True, 
                     show_progress: bool = True) -> List[str]:
    """
    Aplica pré-processamento a uma lista de textos.
    
    Args:
        texts: Lista de textos
        apply_cleaning: Se True, aplica limpeza
        apply_lemmatization: Se True, aplica lematização
        show_progress: Se True, mostra progresso
        
    Returns:
        Lista de textos pré-processados
    """
    if show_progress:
        logger.info(f"Pré-processando {len(texts)} textos...")
        logger.info(f"  - Limpeza: {'Sim' if apply_cleaning else 'Não'}")
        logger.info(f"  - Lematização: {'Sim' if apply_lemmatization else 'Não'}")
    
    # Carrega NLTK uma vez antes do loop
    if apply_lemmatization:
        _load_nltk()
    
    results = []
    for i, text in enumerate(texts):
        processed = preprocess_text(text, apply_cleaning, apply_lemmatization)
        results.append(processed)
        
        if show_progress and (i + 1) % 500 == 0:
            logger.info(f"  Processado: {i + 1}/{len(texts)}")
    
    if show_progress:
        logger.info(f"  Pré-processamento concluído!")
    
    return results


# Exemplo de uso
if __name__ == "__main__":
    # Testa o pré-processamento
    exemplos = [
        "Olá, como você está? #bomdia @usuario",
        "Os gatos estavam correndo pelo jardim rapidamente!",
        "Isso é MUITO ruim!!! https://exemplo.com"
    ]
    
    print("=== Teste de Pré-processamento (NLTK RSLP) ===\n")
    for texto in exemplos:
        print(f"Original: {texto}")
        limpo = clean_text(texto)
        print(f"Limpo: {limpo}")
        lematizado = lemmatize_text(limpo)
        print(f"Lematizado: {lematizado}")
        print()
