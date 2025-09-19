# Manual de Implementação - YouTube Safe Kids

## Visão Geral do Projeto

O **YouTube Safe Kids** é um sistema de filtragem de conteúdo que utiliza modelos de Machine Learning baseados no BERTimbau para classificar comentários do YouTube em diferentes categorias de segurança para crianças.

### Estrutura do Projeto

```
YouTubeSafeKids-Python/
├── app/
│   ├── api/
│   │   └── endpoints/       # Endpoints da API (videos.py)
│   ├── core/
│   │   ├── config.py        # Configurações principais
│   │   └── youtube.py       # Integração com YouTube API
│   ├── nlp/
│   │   ├── models/          # Seus modelos BERTimbau
│   │   ├── datasets/        # Dataset corpus.csv
│   │   ├── training/        # Scripts de treinamento
│   │   ├── evaluation/      # Scripts de avaliação
│   │   ├── utils/          # Utilitários comuns
│   │   └── config/         # Configurações
│   ├── filters/            # Implementação dos filtros
│   ├── static/             # Arquivos estáticos (CSS, JS)
│   └── templates/          # Templates HTML
```

## Divisão de Tarefas

Cada aluno será responsável por implementar **UM** dos seguintes filtros:

1. **Análise de Sentimentos (AS)** - `bertimbau_sentiment.py`
2. **Detecção de Toxicidade (TOX)** - `bertimbau_toxicity.py`
3. **Linguagem Imprópria (LI)** - `bertimbau_language.py`
4. **Tópicos Educacionais (TE)** - `bertimbau_educational.py`

---

## PASSO 1: Configuração do Ambiente

### 1.1 Clone e Branch
```bash
# Clone o repositório (se ainda não fez)
git clone https://github.com/seu-usuario/YouTubeSafeKids-Python.git
cd YouTubeSafeKids-Python

# Crie sua branch específica
git checkout -b feature/filtro-[SEU_FILTRO]
# Exemplo: git checkout -b feature/filtro-sentiment
```

### 1.2 Instale as Dependências
```bash
pip install -r requirements.txt
```

### 1.3 Configure as Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto com as seguintes configurações:
```bash
# YouTube API Configuration
YOUTUBE_API_KEY=sua_chave_da_api_do_youtube_aqui

# Search Configuration  
MAX_SEARCH_RESULTS=50

# Transcription Configuration
ENABLE_VIDEO_TRANSCRIPTION=True
```

**Importante**: Para obter a `YOUTUBE_API_KEY`:
1. Acesse o [Google Cloud Console](https://console.cloud.google.com/)
2. Crie um projeto ou selecione um existente
3. Ative a YouTube Data API v3
4. Gere uma chave de API

### 1.4 Verifique o Dataset
- O dataset está em: `app/nlp/datasets/corpus.csv`
- Contém colunas: `text`, `sentiment`, `toxicity`, `language`, `educational`
- Você usará apenas a coluna correspondente ao seu filtro

---

## PASSO 2: Análise do Dataset

### 2.1 Explore Seu Dataset
```python
import pandas as pd

# Carregue o dataset
df = pd.read_csv('app/nlp/datasets/corpus.csv')

# Para Análise de Sentimentos - use coluna 'sentiment'
# Para Toxicidade - use coluna 'toxicity'  
# Para Linguagem Imprópria - use coluna 'language'
# Para Tópicos Educacionais - use coluna 'educational'

# Exemplo para Sentiment:
print("Distribuição das classes:")
print(df['sentiment'].value_counts())

print("\nExemplos de textos:")
print(df[['text', 'sentiment']].head())
```

### 2.2 Entenda as Classes
- **Sentiment**: 0=Negativo, 1=Neutro, 2=Positivo
- **Toxicity**: 0=Não Tóxico, 1=Levemente Tóxico, 2=Moderadamente Tóxico, 3=Altamente Tóxico
- **Language**: 0=Nenhuma, 1=Leve, 2=Severa
- **Educational**: 0=Não Educacional, 1=Parcialmente Educacional, 2=Educacional

---

## PASSO 3: Implementação do Modelo

### 3.1 Localize Seu Arquivo Template
Seu arquivo está em `app/nlp/models/bertimbau_[SEU_FILTRO].py`

### 3.2 Implemente os Métodos TODO

#### A. Método de Pré-processamento
```python
def preprocess_for_[SEU_FILTRO](self, text: str) -> str:
    """
    Implemente pré-processamento específico para seu domínio.
    
    Exemplos:
    - Sentiment: normalizar emoticons, tratar negações
    - Toxicity: mascarar palavrões, normalizar repetições
    - Language: normalizar gírias, detectar disfarces
    - Educational: identificar termos técnicos, conceitos
    """
    # TODO: Sua implementação aqui
    processed_text = text
    
    # Exemplo de implementações:
    # processed_text = self._normalize_emoticons(text)
    # processed_text = self._handle_negations(processed_text)
    
    return processed_text
```

#### B. Método de Interpretação
```python
def _interpret_[SEU_FILTRO](self, predicted_class: int) -> Dict[str, Any]:
    """
    Interprete a classe predita em termos específicos do seu domínio.
    """
    # TODO: Customize as interpretações para suas classes
    interpretations = {
        0: {
            'level': 'Classe 0',
            'description': 'Descrição da classe 0',
            'recommendation': 'Recomendação para classe 0'
        },
        # ... adicione todas as classes
    }
    
    return interpretations.get(predicted_class, {})
```

#### C. Métodos Auxiliares
Implemente os métodos auxiliares marcados com TODO conforme necessário para seu domínio.

---

## PASSO 4: Preparação dos Dados

### 4.1 Script de Preparação
Crie um script para preparar seus dados **na pasta `app/nlp/datasets/`**:

```python
# prepare_data_[SEU_FILTRO].py
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Carrega dataset
    df = pd.read_csv('app/nlp/datasets/corpus.csv')
    
    # Seleciona colunas relevantes (substitua 'sentiment' pela sua coluna)
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()  # Mude para sua coluna
    
    # Divisão: 80% treino, 10% validação, 10% teste
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

if __name__ == "__main__":
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_data()
    
    print(f"Treino: {len(train_texts)} amostras")
    print(f"Validação: {len(val_texts)} amostras") 
    print(f"Teste: {len(test_texts)} amostras")
```

---

## PASSO 5: Treinamento do Modelo

### 5.1 Script de Treinamento
Crie o script de treinamento **na pasta `app/nlp/training/`**:

```python
# train_[SEU_FILTRO].py
import logging
from app.nlp.models.bertimbau_[SEU_FILTRO] import Bertimbau[SeuFiltro]
from app.nlp.datasets.prepare_data_[SEU_FILTRO] import prepare_data

# Configuração de logging
logging.basicConfig(level=logging.INFO)

def main():
    # Prepara dados
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_data()
    
    # Cria modelo
    model = Bertimbau[SeuFiltro]()  # Substitua pelo nome da sua classe
    
    # Treina modelo
    results = model.train_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        config_name='default',
        experiment_name='[seu_filtro]_v1'
    )
    
    print("Treinamento concluído!")
    print(f"Modelo salvo em: {results['model_path']}")
    print(f"Métricas finais: {results['final_metrics']}")

if __name__ == "__main__":
    main()
```

### 5.2 Execute o Treinamento
```bash
cd app/nlp/training
python train_[SEU_FILTRO].py
```

---

## PASSO 6: Avaliação do Modelo

### 6.1 Script de Avaliação
Crie o script de avaliação **na pasta `app/nlp/evaluation/`**:

```python
# evaluate_[SEU_FILTRO].py
from app.nlp.models.bertimbau_[SEU_FILTRO] import Bertimbau[SeuFiltro]
from app.nlp.evaluation.model_evaluator import ModelEvaluator
from app.nlp.datasets.prepare_data_[SEU_FILTRO] import prepare_data

def main():
    # Prepara dados de teste
    _, _, _, _, test_texts, test_labels = prepare_data()
    
    # Carrega modelo treinado
    model_path = "caminho/para/seu/modelo/treinado"  # Ajuste conforme necessário
    model = Bertimbau[SeuFiltro](model_path=model_path)
    
    # Cria avaliador
    evaluator = ModelEvaluator(task_name='[SEU_FILTRO]')
    
    # Avalia modelo
    results = evaluator.evaluate_model(
        model=model,
        test_texts=test_texts,
        test_labels=test_labels,
        save_results=True,
        experiment_name='[seu_filtro]_evaluation'
    )
    
    print("Avaliação concluída!")
    print(f"Acurácia: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
```

---

## PASSO 7: Testes e Validação

### 7.1 Teste Seu Modelo
Crie o script de teste **na pasta `tests/` (crie a pasta se não existir) ou `app/nlp/utils/`**:

```python
# test_[SEU_FILTRO].py
from app.nlp.models.bertimbau_[SEU_FILTRO] import Bertimbau[SeuFiltro]

def test_model():
    # Carrega modelo
    model = Bertimbau[SeuFiltro](model_path="caminho/para/modelo")
    
    # Textos de teste
    test_texts = [
        "Exemplo de texto 1",
        "Exemplo de texto 2", 
        "Exemplo de texto 3"
    ]
    
    # Testa predições individuais
    for text in test_texts:
        result = model.predict_[seu_metodo_principal](text)
        print(f"Texto: {text}")
        print(f"Predição: {result}")
        print("-" * 50)
    
    # Testa predição em lote
    batch_results = model.analyze_[seu_filtro]_batch(test_texts)
    print(f"Resultados em lote: {len(batch_results)} predições")

if __name__ == "__main__":
    test_model()
```

---

## PASSO 8: Implementação do Filtro

### 8.1 Implemente o Filtro Final
Após treinar e validar seu modelo, implemente o filtro em `app/filters/[seu_filtro].py`:

```python
# app/filters/[seu_filtro].py
from typing import Dict, Any, List
from ..nlp.models.bertimbau_[SEU_FILTRO] import Bertimbau[SeuFiltro]

class [SeuFiltro]Filter:
    """
    Filtro de [Seu Filtro] para o YouTube Safe Kids.
    """
    
    def __init__(self, model_path: str = None):
        self.model = Bertimbau[SeuFiltro](model_path=model_path)
        self.filter_name = "[SEU_FILTRO]"
    
    def filter_comment(self, comment: str) -> Dict[str, Any]:
        """
        Filtra um comentário individual.
        """
        result = self.model.predict_[seu_metodo_principal](comment)
        
        return {
            'filter_name': self.filter_name,
            'is_safe': self._is_safe(result),
            'confidence': result['confidence'],
            'details': result
        }
    
    def filter_comments_batch(self, comments: List[str]) -> List[Dict[str, Any]]:
        """
        Filtra múltiplos comentários.
        """
        results = self.model.analyze_[seu_filtro]_batch(comments)
        
        return [
            {
                'filter_name': self.filter_name,
                'is_safe': self._is_safe(result),
                'confidence': result['confidence'],
                'details': result
            }
            for result in results
        ]
    
    def _is_safe(self, result: Dict[str, Any]) -> bool:
        """
        Determina se o conteúdo é seguro baseado na predição.
        
        TODO: Customize esta lógica para seu filtro específico
        """
        # Exemplo: para sentiment, pode ser seguro se não for muito negativo
        # Para toxicity, seguro se não for tóxico
        # Customize conforme necessário
        predicted_class = result['predicted_class']
        
        # Exemplo de lógica (ajuste para seu caso):
        safe_classes = [0, 1]  # Classes consideradas seguras
        return predicted_class in safe_classes
```

---

## PASSO 9: Documentação e Testes

### 9.1 Documente Seu Trabalho
Crie um arquivo `README_[SEU_FILTRO].md` documentando:
- Abordagem utilizada
- Pré-processamentos aplicados
- Resultados obtidos
- Dificuldades encontradas
- Melhorias futuras

### 9.2 Testes Finais
```python
# final_test_[SEU_FILTRO].py
from app.filters.[seu_filtro] import [SeuFiltro]Filter

def test_filter():
    # Cria filtro
    filter_obj = [SeuFiltro]Filter(model_path="caminho/para/modelo")
    
    # Testa comentários diversos
    test_comments = [
        "Comentário positivo de teste",
        "Comentário negativo de teste",
        "Comentário neutro de teste"
    ]
    
    for comment in test_comments:
        result = filter_obj.filter_comment(comment)
        print(f"Comentário: {comment}")
        print(f"É seguro: {result['is_safe']}")
        print(f"Confiança: {result['confidence']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    test_filter()
```

---

## PASSO 10: Integração e Entrega

### 10.1 Prepare para Merge
```bash
# Adicione todos os arquivos
git add .

# Commit suas mudanças
git commit -m "Implementa filtro de [SEU_FILTRO] com modelo BERTimbau"

# Push para sua branch
git push origin feature/filtro-[SEU_FILTRO]
```

### 10.2 Crie Pull Request
1. Vá para o repositório no GitHub
2. Crie um Pull Request da sua branch para `main`
3. Descreva suas implementações e resultados
4. Aguarde review e aprovação

---

## Dicas Importantes

### ✅ Boas Práticas
- **Sempre teste** seu código antes de fazer commit
- **Documente** suas decisões e implementações
- **Use logging** para acompanhar o treinamento
- **Valide** seus resultados com dados de teste
- **Mantenha** código limpo e comentado

### ⚠️ Cuidados
- **Não modifique** arquivos de outros alunos
- **Não altere** a classe base `BertimbauBase`
- **Não commite** modelos treinados (são muito grandes)
- **Não altere** as configurações principais em `app/core/config.py` sem consultar
- **Não modifique** a integração com YouTube API em `app/core/youtube.py`
- **Teste** em dados pequenos primeiro
- **Monitore** o uso de memória durante treinamento

### 🔧 Troubleshooting
- **Erro de memória**: Reduza batch_size ou max_length
- **Treinamento lento**: Use GPU se disponível
- **Baixa acurácia**: Ajuste pré-processamento ou hiperparâmetros
- **Erro de import**: Verifique PYTHONPATH e estrutura de pastas
- **Erro de API do YouTube**: Verifique se a `YOUTUBE_API_KEY` está configurada corretamente
- **Problemas de transcrição**: Verifique se `ENABLE_VIDEO_TRANSCRIPTION=True` no arquivo `.env`

---

## Recursos Adicionais

### Configurações Disponíveis
- Configurações de modelo: `app/nlp/config/model_config.py`
- Configurações de treinamento: `app/nlp/config/training_config.py`

### Utilitários Disponíveis
- Processamento de dados: `app/nlp/utils/data_utils.py`
- Helpers de treinamento: `app/nlp/utils/training_utils.py`
- Avaliação: `app/nlp/evaluation/model_evaluator.py`

### Contato
- Para dúvidas técnicas: [seu-email@exemplo.com]
- Para problemas de ambiente: [suporte@exemplo.com]

---

**Boa sorte com sua implementação! 🚀**

Lembre-se: o objetivo é criar um filtro eficaz que contribua para tornar o YouTube mais seguro para crianças. Cada filtro é uma peça importante do quebra-cabeça final!