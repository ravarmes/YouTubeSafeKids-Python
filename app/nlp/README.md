<h1 align="center">
    <img alt="YouTube Safe Kids" src="../../static/img/logo.svg" width="200" />
</h1>

<h3 align="center">
  YouTube Safe Kids: Busca Segura de Vídeos com Inteligência Artificial
</h3>

<p align="center">Uma plataforma de filtragem inteligente para conteúdo infantil no YouTube</p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/badge/languages-3-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen">
  <img alt="Made with Python" src="https://img.shields.io/badge/made%20with-Python-blue">
  <img alt="Project Status" src="https://img.shields.io/badge/status-em%20desenvolvimento-yellow">
</p>

## 📑 Sobre o Projeto

O YouTube Safe Kids é uma plataforma avançada que utiliza múltiplas técnicas de Inteligência Artificial para filtrar e recomendar apenas conteúdo adequado e seguro para crianças. O sistema implementa diversos filtros baseados em diferentes tecnologias, desde Processamento de Linguagem Natural (PLN) até Visão Computacional e Machine Learning, garantindo uma experiência segura e educativa.

## 🛠️ Tecnologias Utilizadas

O sistema utiliza uma combinação de diferentes tecnologias de IA para análise de conteúdo:

### Filtros e Tecnologias

| Filtro | Tecnologia | Descrição |
|--------|------------|-----------|
| Sentimento | PLN | Análise do tom emocional do conteúdo |
| Toxicidade | PLN | Detecção de conteúdo tóxico ou ofensivo |
| Educacional | PLN | Avaliação do valor educacional |
| Linguagem | PLN | Identificação de linguagem imprópria |
| Faixa Etária | Machine Learning | Classificação por idade apropriada |
| Diversidade | Visão Computacional | Análise de diversidade visual |
| Duração | Metadados | Filtro baseado na duração do vídeo |
| Engajamento | Metadados | Análise de métricas de engajamento |
| Interatividade | Machine Learning | Avaliação do nível de interatividade |
| Conteúdo Sensível | Visão Computacional | Detecção de conteúdo visual impróprio |

## 🧠 Módulo de Processamento de Linguagem Natural (PLN)

Este módulo contém implementações de modelos PLN baseados no BERTimbau para análise de conteúdo em português brasileiro. Os modelos são utilizados em filtros avançados para garantir que apenas conteúdo adequado e educacional seja recomendado para crianças.

### 📁 Estrutura

```
app/nlp/
├── datasets/             # Conjuntos de dados para treinar os modelos
├── evaluation/           # Scripts para avaliar os modelos
├── models/               # Implementações dos modelos e modelos treinados
│   ├── bertimbau_base.py         # Classe base para os modelos BERTimbau
│   ├── bertimbau_sentiment.py    # Modelo para análise de sentimento
│   ├── bertimbau_toxicity.py     # Modelo para detecção de toxicidade
│   ├── bertimbau_educational.py  # Modelo para classificação educacional
│   └── bertimbau_language.py     # Modelo para detecção de linguagem imprópria
├── training/             # Scripts para treinamento dos modelos
│   └── train_model.py            # Script principal de treinamento
└── utils/                # Utilitários para processamento de texto e dados
```

### 🤖 Modelos PLN Disponíveis

#### 1. Análise de Sentimento (SentimentModel)
- **Descrição**: Classifica textos em sentimentos positivos, negativos ou neutros.
- **Classes**: positivo, negativo, neutro
- **Arquivo**: `models/bertimbau_sentiment.py`

#### 2. Detecção de Toxicidade (ToxicityModel)
- **Descrição**: Identifica conteúdo tóxico, ofensivo ou inadequado para crianças.
- **Classes**: tóxico, não-tóxico
- **Arquivo**: `models/bertimbau_toxicity.py`

#### 3. Classificação Educacional (EducationalModel)
- **Descrição**: Avalia o valor educacional de textos, classificando-os em diferentes níveis.
- **Classes**: não-educacional, baixo_valor_educacional, médio_valor_educacional, alto_valor_educacional
- **Arquivo**: `models/bertimbau_educational.py`

#### 4. Detecção de Linguagem Imprópria (LanguageModel)
- **Descrição**: Identifica linguagem inadequada para crianças, incluindo palavrões e termos inapropriados.
- **Classes**: linguagem_apropriada, linguagem_questionável, linguagem_imprópria
- **Arquivo**: `models/bertimbau_language.py`

## 🔍 Uso dos Modelos PLN

### Preparação de Datasets

1. Colete dados etiquetados para cada tarefa e salve como CSV ou Excel na pasta `datasets/`.
2. Os datasets devem ter pelo menos duas colunas: texto e rótulo.

Exemplos de fontes de datasets em português:
- [Brazilian Portuguese Sentiment Analysis Datasets](https://github.com/pauloemmilio/dataset-sentiment-ptbr)
- [GoEmotions-PT](https://github.com/pratikac/goemotions/tree/master/data)
- [ToxicBR](https://github.com/LaCAfe/Dataset-for-PT-BR) (para toxicidade)

### Treinamento dos Modelos

Execute o script de treinamento para uma tarefa específica:

```bash
python -m app.nlp.training.train_model \
    --task sentiment \
    --data_path app/nlp/datasets/sentiment_data.csv \
    --text_column text \
    --label_column label \
    --output_dir app/nlp/models \
    --epochs 5 \
    --batch_size 16
```

Parâmetros disponíveis:
- `--task`: Tarefa a ser treinada (sentiment, toxicity, educational, language)
- `--data_path`: Caminho para o arquivo de dados (CSV ou Excel)
- `--text_column`: Nome da coluna que contém o texto
- `--label_column`: Nome da coluna que contém os rótulos
- `--output_dir`: Diretório para salvar o modelo treinado
- `--epochs`: Número de épocas de treinamento
- `--batch_size`: Tamanho do batch
- `--learning_rate`: Taxa de aprendizado
- `--max_length`: Tamanho máximo da sequência de tokens

### Uso dos Modelos Treinados

Exemplo de como utilizar um modelo treinado:

```python
from app.nlp.models.bertimbau_sentiment import SentimentModel

# Carregar modelo treinado
model = SentimentModel.from_pretrained("app/nlp/models/bertimbau_sentiment")

# Fazer predição
result = model.predict("Este vídeo é muito divertido e educativo!")
print(f"Sentimento: {result['label']} (confiança: {result['confidence']:.2f})")

# Predição em lote
texts = ["Este vídeo é assustador", "Aprendi muito com esta aula", "Não gostei deste desenho"]
results = model.predict(texts)
for i, res in enumerate(results["results"]):
    print(f"Texto {i+1}: {res['label']} (confiança: {res['confidence']:.2f})")
```

## 🔄 Integração com os Filtros

Os modelos de IA são utilizados pelos filtros correspondentes no sistema:

- **Modelos PLN**:
  - `SentimentModel` é usado pelo `SentimentFilter`
  - `ToxicityModel` é usado pelo `ToxicityFilter`
  - `EducationalModel` é usado pelo `EducationalFilter`
  - `LanguageModel` é usado pelo `LanguageFilter`

- **Modelos de Machine Learning**:
  - Classificadores de idade são usados pelo `AgeRatingFilter`
  - Análise de comportamento é usada pelo `InteractivityFilter`

- **Modelos de Visão Computacional**:
  - Análise de imagem é usada pelo `DiversityFilter`
  - Detecção de conteúdo impróprio é usada pelo `SensitiveFilter`

- **Análise de Metadados**:
  - Processamento de duração é usado pelo `DurationFilter`
  - Análise de engajamento é usada pelo `EngagementFilter`

## ⚙️ Requisitos Técnicos

Este módulo depende dos seguintes pacotes:
- transformers
- torch
- pandas
- numpy
- scikit-learn
- datasets
- opencv-python (para os filtros de visão computacional)
- fastapi (para a API)

Instale os requisitos com:
```bash
pip install transformers torch pandas numpy scikit-learn datasets opencv-python fastapi
```

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/YouTubeSafeKids-Python.git
```

2. Acesse o diretório do projeto:
```bash
cd YouTubeSafeKids-Python
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Inicie o servidor:
```bash
uvicorn app.main:app --reload
```

O aplicativo estará disponível em [http://localhost:8000](http://localhost:8000).

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. 