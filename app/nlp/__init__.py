"""
Módulo de Processamento de Linguagem Natural (PLN) para o YouTube Safe Kids.

Este módulo contém implementações de modelos de PLN baseados no BERTimbau para
análise de conteúdo em português brasileiro, incluindo:
- Análise de sentimento
- Detecção de toxicidade
- Classificação educacional
- Detecção de linguagem imprópria
"""

import logging

logger = logging.getLogger(__name__)
logger.info("Inicializando módulo de NLP") 