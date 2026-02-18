"""
Modelo BERTimbau para Detecção de Toxicidade (TOX).

Este módulo implementa um modelo especializado para detecção de conteúdo tóxico
em comentários do YouTube, utilizando fine-tuning do BERTimbau.

INSTRUÇÕES PARA O ALUNO:
1. Este é um template base - você deve implementar os métodos marcados com TODO
2. Use a classe base BertimbauBase que já fornece funcionalidades comuns
3. Foque na implementação específica para detecção de toxicidade
4. Teste seu modelo com dados de validação antes de finalizar
"""

import os
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
from .bertimbau_base import BertimbauBase
from ..config import get_task_config, get_training_config, PATHS
from ..utils.data_utils import DataProcessor
from ..utils.training_utils import TrainingHelper

logger = logging.getLogger(__name__)

class BertimbauToxicity(BertimbauBase):
    """
    Modelo BERTimbau especializado para Detecção de Toxicidade.
    
    Este modelo classifica textos em 3 níveis de toxicidade:
    - Nenhuma (0)
    - Leve (1)
    - Severa (2)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa o modelo de detecção de toxicidade.
        
        Args:
            model_path: Caminho para modelo pré-treinado (opcional)
            device: Dispositivo para execução (cuda/cpu)
        """
        super().__init__(
            task_name='TOX',
            model_path=model_path,
            device=device
        )
        
        logger.info("Modelo de Detecção de Toxicidade inicializado")
    
    def preprocess_for_toxicity(self, text: str) -> str:
        """
        Pré-processamento específico para detecção de toxicidade.
        
        TODO: Implemente aqui qualquer pré-processamento específico para toxicidade
        Exemplos:
        - Normalização de palavrões mascarados (ex: f*ck -> [PALAVRAO])
        - Tratamento de caracteres especiais usados para contornar filtros
        - Normalização de repetições excessivas (ex: aaahhhhh -> aah)
        - Detecção de padrões de ofensa
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        # TODO: Implementar pré-processamento específico
        # Por enquanto, retorna o texto original
        processed_text = text
        
        # Exemplo de implementações que você pode fazer:
        # processed_text = self._normalize_masked_profanity(text)
        # processed_text = self._handle_special_chars(processed_text)
        # processed_text = self._normalize_repetitions(processed_text)
        # processed_text = self._detect_offensive_patterns(processed_text)
        
        return processed_text
    
    def predict_toxicity(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Prediz o nível de toxicidade de um texto.
        
        Args:
            text: Texto para análise
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Dict com predição de toxicidade
        """
        # Aplica pré-processamento específico
        processed_text = self.preprocess_for_toxicity(text)
        
        # Usa o método predict da classe base
        result = self.predict(processed_text, return_probabilities)
        
        # Adiciona interpretação específica para toxicidade
        result['toxicity_interpretation'] = self._interpret_toxicity(result['predicted_class'])
        
        return result
    
    def _interpret_toxicity(self, predicted_class: int) -> Dict[str, Any]:
        """
        Interpreta a classe predita em termos de toxicidade.
        
        Args:
            predicted_class: Classe predita (0-2)
            
        Returns:
            Dict com interpretação da toxicidade
        """
        interpretations = {
            0: {
                'toxicity_level': 'Nenhuma',
                'description': 'Conteúdo apropriado e seguro para todas as idades',
                'recommendation': 'Permitir conteúdo',
                'severity': 'low'
            },
            1: {
                'toxicity_level': 'Leve',
                'description': 'Conteúdo com linguagem inadequada leve',
                'recommendation': 'Revisar conteúdo',
                'severity': 'medium'
            },
            2: {
                'toxicity_level': 'Severa',
                'description': 'Conteúdo altamente ofensivo e inadequado',
                'recommendation': 'Bloquear conteúdo',
                'severity': 'critical'
            }
        }
        
        return interpretations.get(predicted_class, {
            'toxicity_level': 'Desconhecido',
            'description': 'Classe não reconhecida',
            'recommendation': 'Revisar manualmente',
            'severity': 'unknown'
        })
    
    def train_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        config_name: str = 'default',
        experiment_name: Optional[str] = None,
        save_artifacts: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Treina o modelo de detecção de toxicidade.
        
        TODO: Customize este método conforme necessário para sua implementação
        
        Args:
            train_texts: Textos de treino
            train_labels: Labels de treino
            val_texts: Textos de validação
            val_labels: Labels de validação
            config_name: Nome da configuração de treinamento
            experiment_name: Nome do experimento
            **kwargs: Hiperparâmetros para sobrescrever a configuração padrão (ex: learning_rate, num_train_epochs)
            
        Returns:
            Dict com resultados do treinamento
        """
        logger.info("Iniciando treinamento do modelo de Detecção de Toxicidade")
        
        training_helper = TrainingHelper(
            task_name=self.task_name,
            model_name=self.model_config['base_model'],
            output_base_dir=os.path.join(PATHS['models_dir'], 'trained')
        )
        
        # Configurações de treinamento
        training_config = get_training_config(config_name)
        
        if kwargs:
            logger.info(f"Sobrescrevendo configuração padrão com: {kwargs}")
            training_config.update(kwargs)
            
            if 'max_length' in kwargs:
                self.max_length = kwargs['max_length']
        # Desativa salvamentos automáticos (checkpoints) sempre; salvamos manualmente só no final
        training_config.update({
            'save_strategy': 'no',
            'load_best_model_at_end': False
        })
                
        output_dir = training_helper.get_output_dir(experiment_name)
        
        # TODO: Aplique pré-processamento específico nos dados de treino
        # train_texts = [self.preprocess_for_toxicity(text) for text in train_texts]
        # val_texts = [self.preprocess_for_toxicity(text) for text in val_texts]
        
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
        
        # Remove parâmetros não aceitos por get_training_args
        # Adicione aqui qualquer outro parâmetro que esteja no config mas não em get_training_args
        ignored_keys = ['early_stopping_patience', 'seed', 'max_length', 'task', 'model_name', 'loss_type', 'focal_gamma', 'label_smoothing', 'class_weights']
        training_config_filtered = {k: v for k, v in training_config.items() 
                                    if k not in ignored_keys}
        
        training_args = training_helper.get_training_args(
            output_dir=output_dir,
            **training_config_filtered
        )
        
        # Treina o modelo
        # Parâmetros de perda avançada
        loss_type = kwargs.pop('loss_type', 'focal')
        focal_gamma = float(kwargs.pop('focal_gamma', 2.0))
        label_smoothing = float(kwargs.pop('label_smoothing', 0.0))
        model, trainer = training_helper.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_labels=self.num_labels,
            training_args=training_args,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=None
        )
        
        # Atualiza o modelo atual
        self.model = model
        
        # Avalia no conjunto de validação
        eval_results = trainer.evaluate()
        
        # Gera predições para matriz de confusão
        logger.info("Gerando predições para matriz de confusão...")
        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Calcula matriz de confusão
        cm = confusion_matrix(labels, preds)
        eval_results['confusion_matrix'] = cm.tolist() # Converte para lista para serialização JSON
        
        if save_artifacts:
            training_helper.save_model_with_metadata(
                model=model,
                tokenizer=self.tokenizer,
                output_dir=output_dir,
                training_args=training_args,
                metrics=eval_results,
                additional_info={
                    'task_specific_info': 'Modelo treinado para detecção de toxicidade',
                    'preprocessing_applied': 'TODO: Descrever pré-processamentos aplicados'
                }
            )
        
        logger.info(f"Treinamento concluído. Modelo salvo em {output_dir}")
        
        return {
            'model_path': output_dir,
            'final_metrics': eval_results,
            'training_config': training_config
        }
    
    def analyze_toxicity_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analisa toxicidade de múltiplos textos.
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do lote
            
        Returns:
            Lista com análises de toxicidade
        """
        # Aplica pré-processamento
        processed_texts = [self.preprocess_for_toxicity(text) for text in texts]
        
        # Usa predição em lote da classe base
        results = self.predict_batch(processed_texts, batch_size)
        
        # Adiciona interpretações específicas
        for result in results:
            result['toxicity_interpretation'] = self._interpret_toxicity(result['predicted_class'])
        
        return results
    
    def get_toxicity_score(self, text: str) -> float:
        """
        Retorna um score de toxicidade entre 0 e 1.
        
        TODO: Implemente um sistema de scoring personalizado
        
        Args:
            text: Texto para análise
            
        Returns:
            Score de toxicidade (0 = não tóxico, 1 = altamente tóxico)
        """
        result = self.predict_toxicity(text, return_probabilities=True)
        
        # TODO: Customize este cálculo baseado em suas necessidades
        # Exemplo simples: usa a probabilidade da classe mais alta
        if 'probabilities' in result:
            weights_by_class = {
                'Nenhuma': 0.0,
                'Leve': 0.5,
                'Severa': 1.0
            }
            probs = result['probabilities']
            if isinstance(probs, dict):
                score = sum(probs.get(cls, 0.0) * weights_by_class.get(cls, 0.0) for cls in self.class_names)
            else:
                weights = [weights_by_class.get(cls, 0.0) for cls in self.class_names]
                score = sum(prob * weight for prob, weight in zip(probs, weights))
            return min(max(score, 0.0), 1.0)
        
        # Fallback: usa apenas a classe predita
        return result['predicted_class'] / 2.0
    
    # TODO: Implemente métodos auxiliares conforme necessário
    def _normalize_masked_profanity(self, text: str) -> str:
        """
        Normaliza palavrões mascarados.
        
        TODO: Implemente normalização de palavrões mascarados
        Exemplo: f*ck -> [PALAVRAO], sh!t -> [PALAVRAO]
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _handle_special_chars(self, text: str) -> str:
        """
        Trata caracteres especiais usados para contornar filtros.
        
        TODO: Implemente tratamento de caracteres especiais
        Exemplo: @ -> a, 3 -> e, 1 -> i
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _normalize_repetitions(self, text: str) -> str:
        """
        Normaliza repetições excessivas de caracteres.
        
        TODO: Implemente normalização de repetições
        Exemplo: aaahhhhh -> aah, noooooo -> noo
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _detect_offensive_patterns(self, text: str) -> str:
        """
        Detecta e marca padrões ofensivos.
        
        TODO: Implemente detecção de padrões ofensivos
        """
        # Implementação exemplo - você deve expandir isso
        return text


# Função de conveniência para criar e usar o modelo
def create_toxicity_model(model_path: Optional[str] = None) -> BertimbauToxicity:
    """
    Cria uma instância do modelo de detecção de toxicidade.
    
    Args:
        model_path: Caminho para modelo pré-treinado
        
    Returns:
        Instância do modelo
    """
    return BertimbauToxicity(model_path=model_path)


# Exemplo de uso (para testes durante desenvolvimento)
if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(level=logging.INFO)
    
    # Cria modelo
    model = create_toxicity_model()
    
    # Exemplo de uso
    test_text = "Este comentário contém linguagem inadequada!"
    result = model.predict_toxicity(test_text)
    toxicity_score = model.get_toxicity_score(test_text)
    
    print(f"Texto: {test_text}")
    print(f"Toxicidade: {result['predicted_label']}")
    print(f"Confiança: {result['confidence']:.4f}")
    print(f"Score: {toxicity_score:.4f}")
    print(f"Interpretação: {result['toxicity_interpretation']}")
