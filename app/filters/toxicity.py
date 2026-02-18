from .base import BaseFilter
from ..nlp.models.bertimbau_toxicity import BertimbauToxicity
from typing import Dict, Any, List
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Pesos de penalização por classe de toxicidade (adaptado para 3 classes)
# Nenhuma: sem penalização; Leve: até 50%; Severa: até 100%
TOXICITY_WEIGHTS = {
    0: 0.0,   # Nenhuma
    1: 0.5,   # Leve
    2: 1.0,   # Severa
}


class ToxicityFilter(BaseFilter):
    """
    Filtro de toxicidade usando modelo BERTimbau especializado.
    Classifica conteúdo em 3 níveis: Nenhuma (0), Leve (1), Severa (2).
    
    O Score de cada frase é calculado por análise probabilística:
        Pen  = Σ prob_i × w_i   (soma ponderada das probabilidades)
        P    = 1 - Pen           (score da frase)
    O Score final do vídeo é a média dos scores das 3 frases
    representativas (início, meio e fim da transcrição).
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Toxicidade",
            description="Detecta conteúdo tóxico, ofensivo ou inadequado",
            default_enabled=True
        )
        
        # --- LÓGICA AUTOMÁTICA PARA ENCONTRAR O MODELO TREINADO ---
        if model_path is None:
            try:
                current_dir = Path(__file__).parent
                models_dir = current_dir.parent / 'nlp' / 'models' / 'trained'
                
                if models_dir.exists():
                    model_paths = list(models_dir.glob('TOX_*'))
                    if model_paths:
                        latest_model = max(model_paths, key=lambda p: p.stat().st_mtime)
                        model_path = str(latest_model)
                        logger.info(f"Modelo de toxicidade encontrado automaticamente: {model_path}")
                    else:
                        logger.warning("Nenhum modelo treinado 'TOX_*' encontrado na pasta trained.")
                else:
                    logger.warning(f"Diretório de modelos não encontrado: {models_dir}")
            except Exception as e:
                logger.error(f"Erro ao tentar localizar modelo automaticamente: {e}")
        
        try:
            self.model = BertimbauToxicity(model_path=model_path)
            logger.info(f"Modelo de toxicidade carregado com sucesso de: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de toxicidade: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Cálculo do Score probabilístico
    # ------------------------------------------------------------------

    def _compute_sentence_score(self, text: str) -> float:
        """
        Calcula o Score de segurança de uma única frase usando
        análise probabilística com penalização ponderada.

        Pen = Σ prob_i × w_i
        Score = 1 - Pen

        Args:
            text: Frase para análise.

        Returns:
            float: Score entre 0 (muito tóxico) e 1 (seguro).
        """
        if not text or not text.strip():
            return 1.0  # Sem texto → considera seguro

        result = self.model.predict_toxicity(text, return_probabilities=True)

        probabilities = result.get('probabilities', {})

        # Calcula penalização ponderada
        penalty = 0.0
        if isinstance(probabilities, dict):
            # probabilities pode ser {'Nenhuma': 0.9, 'Leve': 0.05, 'Severa': 0.05}
            class_names = self.model.class_names if hasattr(self.model, 'class_names') else []
            for idx, cls_name in enumerate(class_names):
                prob = probabilities.get(cls_name, 0.0)
                weight = TOXICITY_WEIGHTS.get(idx, 0.0)
                penalty += prob * weight
        elif isinstance(probabilities, (list, tuple)):
            for idx, prob in enumerate(probabilities):
                weight = TOXICITY_WEIGHTS.get(idx, 0.0)
                penalty += prob * weight
        else:
            # Fallback: usa classe predita
            predicted_class = result.get('predicted_class', 0)
            confidence = result.get('confidence', 0.5)
            if predicted_class == 0:
                penalty = 0.0
            elif predicted_class == 1:
                penalty = 0.3 * confidence
            else:
                penalty = 1.0 * confidence

        score = 1.0 - penalty
        return min(max(score, 0.0), 1.0)

    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo para detectar toxicidade usando análise probabilística
        de 3 frases representativas (início, meio e fim da transcrição).

        O Score final é a média aritmética dos scores individuais das 3 frases.

        Args:
            video_data: Dados do vídeo incluindo sentences (start, middle, end).

        Returns:
            float: Score de 0 a 1 (0 = muito tóxico, 1 = não tóxico).
        """
        if self.model is None:
            logger.warning("Modelo de toxicidade não disponível, retornando score neutro")
            return 0.5

        # Obtém as 3 frases representativas
        sentences = video_data.get('sentences', {})
        start_text = sentences.get('start', '')
        middle_text = sentences.get('middle', '')
        end_text = sentences.get('end', '')

        # Sem frases → não deveria acontecer (vídeos sem transcrição são filtrados)
        if not any([start_text.strip(), middle_text.strip(), end_text.strip()]):
            logger.warning("Nenhuma frase disponível para análise de toxicidade")
            return 0.5  # Score neutro

        try:
            scores = []
            phrase_labels = ['início', 'meio', 'fim']
            phrase_texts = [start_text, middle_text, end_text]

            for label, text in zip(phrase_labels, phrase_texts):
                if text.strip():
                    score = self._compute_sentence_score(text)
                    scores.append(score)
                    logger.info(f"Toxicidade frase {label}: Score={score:.4f} | '{text[:60]}...'")
                else:
                    logger.info(f"Toxicidade frase {label}: vazia, ignorada")

            if not scores:
                return 1.0

            # Score final = média dos scores das frases
            final_score = sum(scores) / len(scores)
            logger.info(f"Score final de toxicidade: {final_score:.4f} (média de {len(scores)} frases)")

            return min(max(final_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Erro ao processar toxicidade: {e}")
            return 0.5

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de toxicidade.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": "BERTimbau Fine-tuned (TOX)" if self.model else "Modelo não carregado",
            "options": {
                "toxicity_threshold": {
                    "type": "slider",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                    "step": 0.1,
                    "description": "Limiar de toxicidade (0.0=muito restritivo, 1.0=pouco restritivo)"
                }
            }
        }