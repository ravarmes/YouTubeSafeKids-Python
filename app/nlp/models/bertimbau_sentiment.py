"""
Modelo BERTimbau para Análise de Sentimentos (AS).

Este módulo implementa um modelo especializado para classificação de sentimentos
em comentários do YouTube, utilizando fine-tuning do BERTimbau.

INSTRUÇÕES PARA O ALUNO:
1. Este é um template base - você deve implementar os métodos marcados com TODO
2. Use a classe base BertimbauBase que já fornece funcionalidades comuns
3. Foque na implementação específica para análise de sentimentos
4. Teste seu modelo com dados de validação antes de finalizar
"""

import os
import logging
from typing import Dict, List, Any, Optional
from .bertimbau_base import BertimbauBase
from ..config import get_task_config, get_training_config
from ..utils.data_utils import DataProcessor
from ..utils.training_utils import TrainingHelper

logger = logging.getLogger(__name__)

class BertimbauSentiment(BertimbauBase):
    """
    Modelo BERTimbau especializado para Análise de Sentimentos.
    
    Este modelo classifica textos em 3 categorias de sentimento:
    - Negativo (0)
    - Neutro (1) 
    - Positivo (2)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa o modelo de análise de sentimentos.
        
        Args:
            model_path: Caminho para modelo pré-treinado (opcional)
            device: Dispositivo para execução (cuda/cpu)
        """
        super().__init__(
            task_name='AS',
            model_path=model_path,
            device=device
        )
        
        logger.info("Modelo de Análise de Sentimentos inicializado")
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Pré-processamento específico para análise de sentimentos.
        
        TODO: Implemente aqui qualquer pré-processamento específico para sentimentos
        Exemplos:
        - Normalização de emoticons
        - Tratamento de negações
        - Limpeza de caracteres especiais
        - Expansão de contrações
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        # TODO: Implementar pré-processamento específico
        # Por enquanto, retorna o texto original
        processed_text = text
        
        # Exemplo de implementações que você pode fazer:
        # processed_text = self._normalize_emoticons(text)
        # processed_text = self._handle_negations(processed_text)
        # processed_text = self._clean_text(processed_text)
        
        return processed_text
    
    def predict_sentiment(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Prediz o sentimento de um texto.
        
        Args:
            text: Texto para análise
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Dict com predição de sentimento
        """
        # Aplica pré-processamento específico
        processed_text = self.preprocess_for_sentiment(text)
        
        # Usa o método predict da classe base
        result = self.predict(processed_text, return_probabilities)
        
        # Adiciona interpretação específica para sentimentos
        result['sentiment_interpretation'] = self._interpret_sentiment(result['predicted_class'])
        
        return result
    
    def _interpret_sentiment(self, predicted_class: int) -> Dict[str, Any]:
        """
        Interpreta a classe predita em termos de sentimento.
        
        TODO: Implemente interpretações específicas para seu domínio
        
        Args:
            predicted_class: Classe predita (0-2)
            
        Returns:
            Dict com interpretação do sentimento
        """
        interpretations = {
            0: {
                'sentiment': 'Negativo',
                'description': 'Conteúdo com sentimento negativo, pode ser inadequado para crianças',
                'recommendation': 'Revisar conteúdo'
            },
            1: {
                'sentiment': 'Neutro',
                'description': 'Conteúdo com sentimento neutro',
                'recommendation': 'Permitir conteúdo'
            },
            2: {
                'sentiment': 'Positivo',
                'description': 'Conteúdo com sentimento positivo',
                'recommendation': 'Permitir conteúdo'
            }
        }
        
        return interpretations.get(predicted_class, {
            'sentiment': 'Desconhecido',
            'description': 'Classe não reconhecida',
            'recommendation': 'Revisar manualmente'
        })
    
    def train_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        config_name: str = 'default',
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Treina o modelo de análise de sentimentos.
        
        TODO: Customize este método conforme necessário para sua implementação
        
        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            val_texts: Textos de validação
            val_labels: Labels de validação
            config_name: Nome da configuração de treinamento
            experiment_name: Nome do experimento
            
        Returns:
            Dict com resultados do treinamento
        """
        logger.info("Iniciando treinamento do modelo de Análise de Sentimentos")
        
        # Cria helper de treinamento
        training_helper = TrainingHelper(
            task_name=self.task_name,
            model_name=self.model_config['base_model']
        )
        
        # TODO: Aplique pré-processamento específico nos dados de treino
        # train_texts = [self.preprocess_for_sentiment(text) for text in train_texts]
        # val_texts = [self.preprocess_for_sentiment(text) for text in val_texts]
        
        # Prepara datasets
        train_dataset, val_dataset, _ = training_helper.prepare_datasets(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            test_texts=[],  # Não usado no treinamento
            test_labels=[],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Configurações de treinamento
        training_config = get_training_config(config_name)
        output_dir = training_helper.get_output_dir(experiment_name)
        
        training_args = training_helper.get_training_args(
            output_dir=output_dir,
            **training_config
        )
        
        # Treina o modelo
        model, trainer = training_helper.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_labels=self.num_labels,
            training_args=training_args
        )
        
        # Atualiza o modelo atual
        self.model = model
        
        # Avalia no conjunto de validação
        eval_results = trainer.evaluate()
        
        # Salva o modelo
        training_helper.save_model_with_metadata(
            model=model,
            tokenizer=self.tokenizer,
            output_dir=output_dir,
            training_args=training_args,
            metrics=eval_results,
            additional_info={
                'task_specific_info': 'Modelo treinado para análise de sentimentos',
                'preprocessing_applied': 'TODO: Descrever pré-processamentos aplicados'
            }
        )
        
        logger.info(f"Treinamento concluído. Modelo salvo em {output_dir}")
        
        return {
            'model_path': output_dir,
            'final_metrics': eval_results,
            'training_config': training_config
        }
    
    def analyze_sentiment_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analisa sentimento de múltiplos textos.
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do lote
            
        Returns:
            Lista com análises de sentimento
        """
        # Aplica pré-processamento
        processed_texts = [self.preprocess_for_sentiment(text) for text in texts]
        
        # Usa predição em lote da classe base
        results = self.predict_batch(processed_texts, batch_size)
        
        # Adiciona interpretações específicas
        for result in results:
            result['sentiment_interpretation'] = self._interpret_sentiment(result['predicted_class'])
        
        return results
    
    # TODO: Implemente métodos auxiliares conforme necessário
    def _normalize_emoticons(self, text: str) -> str:
        """
        Normaliza emoticons no texto.
        
        TODO: Implemente normalização de emoticons
        Exemplo: :) -> [EMOTICON_POSITIVO]
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _handle_negations(self, text: str) -> str:
        """
        Trata negações no texto.
        
        TODO: Implemente tratamento de negações
        Exemplo: "não gostei" -> "não_gostei"
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo caracteres desnecessários.
        
        TODO: Implemente limpeza específica para sentimentos
        """
        # Implementação exemplo - você deve expandir isso
        return text.strip()


# Função de conveniência para criar e usar o modelo
def create_sentiment_model(model_path: Optional[str] = None) -> BertimbauSentiment:
    """
    Cria uma instância do modelo de análise de sentimentos.
    
    Args:
        model_path: Caminho para modelo pré-treinado
        
    Returns:
        Instância do modelo
    """
    return BertimbauSentiment(model_path=model_path)


# Exemplo de uso (para testes durante desenvolvimento)
if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(level=logging.INFO)
    
    # Cria modelo
    model = create_sentiment_model()
    
    # Exemplo de uso
    test_text = "Este vídeo é muito interessante e educativo!"
    result = model.predict_sentiment(test_text)
    
    print(f"Texto: {test_text}")
    print(f"Sentimento: {result['predicted_label']}")
    print(f"Confiança: {result['confidence']:.4f}")
    print(f"Interpretação: {result['sentiment_interpretation']}")