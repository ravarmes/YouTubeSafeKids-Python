"""
Módulo de avaliação de modelos.

Este módulo contém classes e funções para avaliar o desempenho
dos modelos treinados.
"""

from .model_evaluator import ModelEvaluator
from .cross_validation import CrossValidator
from .metrics_calculator import MetricsCalculator

__all__ = [
    'ModelEvaluator',
    'CrossValidator', 
    'MetricsCalculator'
]