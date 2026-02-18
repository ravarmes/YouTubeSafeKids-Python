"""
Script de teste para Análise de Sentimentos (AS).

Este script testa o modelo treinado de análise de sentimentos
com exemplos de texto para verificar o funcionamento.
"""

import logging
import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model():
    """Testa o modelo com exemplos de texto."""
    logger.info("=" * 60)
    logger.info("Testando modelo de Análise de Sentimentos")
    logger.info("=" * 60)
    
    # Caminho do modelo treinado (ajuste conforme necessário)
    models_dir = Path(__file__).parent.parent.parent / 'nlp' / 'models' / 'trained'
    
    # Lista modelos disponíveis
    model_paths = list(models_dir.glob('AS_*'))
    if not model_paths:
        logger.warning(f"Nenhum modelo encontrado em {models_dir}")
        logger.info("Usando modelo base (não treinado)...")
        model = BertimbauSentiment()
    else:
        # Usa o modelo mais recente
        model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
        logger.info(f"Carregando modelo de: {model_path}")
        model = BertimbauSentiment(model_path=str(model_path))
    
    # Textos de teste diversos
    test_texts = [
        "Este vídeo é muito interessante e educativo!",
        "Não gostei nada desse conteúdo, muito ruim.",
        "O vídeo explica conceitos básicos de matemática.",
        "Que porcaria! Esse conteúdo não deveria estar aqui.",
        "Aprender é sempre bom, gostei bastante.",
        "Esse vídeo é mediano, nem bom nem ruim."
    ]
    
    logger.info(f"Testando {len(test_texts)} textos...")
    print("\n" + "=" * 60)
    print("RESULTADOS DOS TESTES")
    print("=" * 60)
    
    # Testa predições individuais
    for i, text in enumerate(test_texts, 1):
        result = model.predict_sentiment(text)
        
        print(f"\nTeste {i}:")
        print(f"  Texto: {text}")
        print(f"  Classe predita: {result['predicted_class']} ({result['predicted_label']})")
        print(f"  Confiança: {result['confidence']:.4f}")
        
        if 'sentiment_interpretation' in result:
            interpretation = result['sentiment_interpretation']
            print(f"  Sentimento: {interpretation.get('sentiment', 'N/A')}")
            print(f"  Descrição: {interpretation.get('description', 'N/A')}")
            print(f"  Recomendação: {interpretation.get('recommendation', 'N/A')}")
        
        if 'probabilities' in result:
            print(f"  Probabilidades: {result['probabilities']}")
        
        print("-" * 60)
    
    # Testa predição em lote
    print("\n" + "=" * 60)
    print("TESTE EM LOTE")
    print("=" * 60)
    
    batch_results = model.analyze_sentiment_batch(test_texts)
    print(f"\nProcessados {len(batch_results)} textos em lote")
    
    print("\nResumo das predições:")
    sentiment_counts = {}
    for result in batch_results:
        sentiment = result.get('sentiment_interpretation', {}).get('sentiment', 'Desconhecido')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
    
    print("\n" + "=" * 60)
    logger.info("Testes concluídos!")

if __name__ == "__main__":
    test_model()

