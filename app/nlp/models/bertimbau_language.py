"""
Modelo BERTimbau para Detecção de Linguagem Imprópria (LI).

Este módulo implementa um modelo especializado para detecção de linguagem imprópria
em comentários do YouTube, utilizando fine-tuning do BERTimbau.

INSTRUÇÕES PARA O ALUNO:
1. Este é um template base - você deve implementar os métodos marcados com TODO
2. Use a classe base BertimbauBase que já fornece funcionalidades comuns
3. Foque na implementação específica para detecção de linguagem imprópria
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

class BertimbauLanguage(BertimbauBase):
    """
    Modelo BERTimbau especializado para Detecção de Linguagem Imprópria.
    
    Este modelo classifica textos em 3 níveis de linguagem imprópria:
    - Nenhuma (0)
    - Leve (1)
    - Severa (2)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Inicializa o modelo de detecção de linguagem imprópria.
        
        Args:
            model_path: Caminho para modelo pré-treinado (opcional)
            device: Dispositivo para execução (cuda/cpu)
        """
        super().__init__(
            task_name='LI',
            model_path=model_path,
            device=device
        )
        
        logger.info("Modelo de Detecção de Linguagem Imprópria inicializado")
    
    def preprocess_for_language(self, text: str) -> str:
        """
        Pré-processamento específico para detecção de linguagem imprópria.
        
        TODO: Implemente aqui qualquer pré-processamento específico para linguagem imprópria
        Exemplos:
        - Normalização de gírias e expressões coloquiais
        - Detecção de linguagem vulgar disfarçada
        - Tratamento de abreviações e internetês
        - Identificação de padrões de linguagem inadequada
        
        Args:
            text: Texto original
            
        Returns:
            Texto pré-processado
        """
        # TODO: Implementar pré-processamento específico
        # Por enquanto, retorna o texto original
        processed_text = text
        
        # Exemplo de implementações que você pode fazer:
        # processed_text = self._normalize_slang(text)
        # processed_text = self._detect_disguised_vulgar(processed_text)
        # processed_text = self._handle_abbreviations(processed_text)
        # processed_text = self._identify_inappropriate_patterns(processed_text)
        
        return processed_text
    
    def predict_language_appropriateness(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Prediz a adequação da linguagem de um texto.
        
        Args:
            text: Texto para análise
            return_probabilities: Se deve retornar probabilidades
            
        Returns:
            Dict com predição de adequação da linguagem
        """
        # Aplica pré-processamento específico
        processed_text = self.preprocess_for_language(text)
        
        # Usa o método predict da classe base
        result = self.predict(processed_text, return_probabilities)
        
        # Adiciona interpretação específica para linguagem
        result['language_interpretation'] = self._interpret_language(result['predicted_class'])
        
        return result
    
    def _interpret_language(self, predicted_class: int) -> Dict[str, Any]:
        """
        Interpreta a classe predita em termos de adequação da linguagem.
        
        TODO: Implemente interpretações específicas para seu domínio
        
        Args:
            predicted_class: Classe predita (0-2)
            
        Returns:
            Dict com interpretação da adequação da linguagem
        """
        interpretations = {
            0: {
                'language_level': 'Nenhuma',
                'description': 'Linguagem completamente apropriada para todas as idades',
                'recommendation': 'Permitir conteúdo',
                'severity': 'none',
                'action': 'allow'
            },
            1: {
                'language_level': 'Leve',
                'description': 'Linguagem com elementos levemente questionáveis',
                'recommendation': 'Revisar conteúdo para contexto',
                'severity': 'low',
                'action': 'review'
            },
            2: {
                'language_level': 'Severa',
                'description': 'Linguagem imprópria que pode ser inadequada para crianças',
                'recommendation': 'Bloquear ou filtrar conteúdo',
                'severity': 'high',
                'action': 'block'
            }
        }
        
        return interpretations.get(predicted_class, {
            'language_level': 'Desconhecido',
            'description': 'Classe não reconhecida',
            'recommendation': 'Revisar manualmente',
            'age_appropriateness': 'unknown'
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
        Treina o modelo de detecção de linguagem imprópria.
        
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
        logger.info("Iniciando treinamento do modelo de Detecção de Linguagem Imprópria")
        
        # Cria helper de treinamento
        training_helper = TrainingHelper(
            task_name=self.task_name,
            model_name=self.model_config['base_model']
        )
        
        # TODO: Aplique pré-processamento específico nos dados de treino
        # train_texts = [self.preprocess_for_language(text) for text in train_texts]
        # val_texts = [self.preprocess_for_language(text) for text in val_texts]
        
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
                'task_specific_info': 'Modelo treinado para detecção de linguagem imprópria',
                'preprocessing_applied': 'TODO: Descrever pré-processamentos aplicados'
            }
        )
        
        logger.info(f"Treinamento concluído. Modelo salvo em {output_dir}")
        
        return {
            'model_path': output_dir,
            'final_metrics': eval_results,
            'training_config': training_config
        }
    
    def analyze_language_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Analisa adequação da linguagem de múltiplos textos.
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do lote
            
        Returns:
            Lista com análises de linguagem
        """
        # Aplica pré-processamento
        processed_texts = [self.preprocess_for_language(text) for text in texts]
        
        # Usa predição em lote da classe base
        results = self.predict_batch(processed_texts, batch_size)
        
        # Adiciona interpretações específicas
        for result in results:
            result['language_interpretation'] = self._interpret_language(result['predicted_class'])
        
        return results
    
    def get_appropriateness_score(self, text: str) -> float:
        """
        Retorna um score de adequação da linguagem entre 0 e 1.
        
        Args:
            text: Texto para análise
            
        Returns:
            Score de adequação (0 = severa, 1 = nenhuma)
        """
        result = self.predict_language_appropriateness(text, return_probabilities=True)
        
        # Pesos para cada classe: quanto mais apropriado, maior o score
        class_weights = {
            'Nenhuma': 1.0,
            'Leve': 0.5,
            'Severa': 0.0
        }
        
        if 'probabilities' in result:
            # Calcula score ponderado baseado nas probabilidades
            score = sum(
                result['probabilities'].get(label, 0) * weight 
                for label, weight in class_weights.items()
            )
            return min(max(score, 0.0), 1.0)  # Garante que está entre 0 e 1
        
        # Fallback: usa apenas a classe predita (invertida)
        return 1.0 - (result['predicted_class'] / 2.0)
    
    def is_age_appropriate(self, text: str, min_age: int = 13) -> Dict[str, Any]:
        """
        Verifica se o texto tem linguagem apropriada para uma idade específica.
        
        TODO: Implemente lógica específica para diferentes idades
        
        Args:
            text: Texto para análise
            min_age: Idade mínima para considerar linguagem apropriada
            
        Returns:
            Dict com informações sobre adequação por idade
        """
        result = self.predict_language_appropriateness(text)
        interpretation = result['language_interpretation']
        
        # TODO: Customize esta lógica baseada em suas necessidades
        age_mapping = {
            'all_ages': 0,
            'teen_plus': 13,
            'adult_only': 18,
            'restricted': 21,
            'unknown': 18  # Conservador para casos desconhecidos
        }
        
        required_age = age_mapping.get(interpretation['age_appropriateness'], 18)
        is_appropriate = min_age >= required_age
        
        return {
            'is_appropriate': is_appropriate,
            'required_age': required_age,
            'provided_age': min_age,
            'language_level': interpretation['language_level'],
            'recommendation': interpretation['recommendation']
        }
    
    # TODO: Implemente métodos auxiliares conforme necessário
    def _normalize_slang(self, text: str) -> str:
        """
        Normaliza gírias e expressões coloquiais.
        
        TODO: Implemente normalização de gírias
        Exemplo: "mano" -> "cara", "trampo" -> "trabalho"
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _detect_disguised_vulgar(self, text: str) -> str:
        """
        Detecta linguagem vulgar disfarçada.
        
        TODO: Implemente detecção de linguagem vulgar disfarçada
        Exemplo: "p0rr@" -> "[VULGAR]"
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _handle_abbreviations(self, text: str) -> str:
        """
        Trata abreviações e internetês.
        
        TODO: Implemente tratamento de abreviações
        Exemplo: "vc" -> "você", "pq" -> "porque"
        """
        # Implementação exemplo - você deve expandir isso
        return text
    
    def _identify_inappropriate_patterns(self, text: str) -> str:
        """
        Identifica padrões de linguagem inadequada.
        
        TODO: Implemente identificação de padrões inadequados
        """
        # Implementação exemplo - você deve expandir isso
        return text


# Função de conveniência para criar e usar o modelo
def create_language_model(model_path: Optional[str] = None) -> BertimbauLanguage:
    """
    Cria uma instância do modelo de detecção de linguagem imprópria.
    
    Args:
        model_path: Caminho para modelo pré-treinado
        
    Returns:
        Instância do modelo
    """
    return BertimbauLanguage(model_path=model_path)


# Exemplo de uso (para testes durante desenvolvimento)
if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(level=logging.INFO)
    
    # Cria modelo
    model = create_language_model()
    
    # Exemplo de uso
    test_text = "Esse vídeo é muito legal, cara!"
    result = model.predict_language_appropriateness(test_text)
    appropriateness_score = model.get_appropriateness_score(test_text)
    age_check = model.is_age_appropriate(test_text, min_age=10)
    
    print(f"Texto: {test_text}")
    print(f"Linguagem: {result['predicted_label']}")
    print(f"Confiança: {result['confidence']:.4f}")
    print(f"Score de adequação: {appropriateness_score:.4f}")
    print(f"Linguagem apropriada para 10 anos: {age_check['is_appropriate']}")
    print(f"Interpretação: {result['language_interpretation']}")