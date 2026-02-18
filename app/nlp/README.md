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
- **Classes**: 0=Negativo, 1=Neutro, 2=Positivo
- **Arquivo**: `models/bertimbau_sentiment.py`

#### 2. Detecção de Toxicidade (ToxicityModel)
- **Descrição**: Identifica conteúdo tóxico, ofensivo ou inadequado para crianças.
- **Classes**: 0=Não Tóxico, 1=Levemente Tóxico, 2=Moderadamente Tóxico, 3=Altamente Tóxico
- **Arquivo**: `models/bertimbau_toxicity.py`

#### 3. Classificação Educacional (EducationalModel)
- **Descrição**: Avalia o valor educacional de textos, classificando-os em diferentes níveis.
- **Classes**: 0=Não Educacional, 1=Potencialmente Educacional, 2=Educacional, 3=Altamente Educacional
- **Arquivo**: `models/bertimbau_educational.py`

#### 4. Detecção de Linguagem Imprópria (LanguageModel)
- **Descrição**: Identifica linguagem inadequada para crianças, incluindo palavrões e termos inapropriados.
- **Classes**: 0=Apropriado, 1=Questionável, 2=Inapropriado, 3=Altamente Inapropriado
- **Arquivo**: `models/bertimbau_language.py`

## 🔍 Uso dos Modelos PLN

### Preparação de Datasets

1. Colete dados etiquetados para cada tarefa e salve como CSV ou Excel na pasta `datasets/`.
2. Os datasets devem ter pelo menos duas colunas: texto e rótulo.

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

## 🎥 Integração com YouTube API

O sistema utiliza a YouTube Data API v3 para buscar e analisar vídeos. A integração é feita através do módulo `app/core/youtube.py` que é utilizado pelo endpoint de busca em `app/api/endpoints/videos.py`.

### Fluxo de Busca e Análise

1. **Busca de Vídeos**: O endpoint `/search/` recebe uma consulta e parâmetros de filtros
2. **Chamada da API**: O `YouTubeAPI.search_videos()` busca vídeos usando a YouTube Data API
3. **Obtenção de Transcrições**: Se filtros PLN estão habilitados, o sistema obtém transcrições dos vídeos
4. **Processamento**: Cada vídeo é processado pelos filtros habilitados
5. **Classificação**: Os vídeos são ordenados por score final e retornados

### Método de Pesquisa (`search_videos`)

```python
async def search_videos(self, query: str, max_results: int = None, 
                       video_duration: str = None, 
                       nlp_filters_enabled: List[str] = None) -> List[Dict[str, Any]]
```

**Parâmetros:**
- `query`: Termo de busca
- `max_results`: Número máximo de resultados (padrão: `MAX_SEARCH_RESULTS`)
- `video_duration`: Filtro de duração ('short', 'medium', 'long')
- `nlp_filters_enabled`: Lista de filtros PLN que requerem transcrição

**Funcionalidades:**
- Busca com `safeSearch: strict` para conteúdo apropriado para crianças
- Filtragem por duração quando especificada
- Obtenção de metadados detalhados (visualizações, likes, comentários)
- Extração de transcrições quando filtros PLN estão ativos

### Sistema de Transcrição

O sistema de transcrição é controlado pela configuração `ENABLE_VIDEO_TRANSCRIPTION` e funciona da seguinte forma:

#### Método `get_video_sentences`

Extrai três frases representativas de cada vídeo:
- **Início**: Primeiros 10% da transcrição
- **Meio**: Segmentos de 40% a 60% da transcrição  
- **Fim**: Últimos 10% da transcrição

```python
async def get_video_sentences(self, video_id: str) -> Dict[str, str]:
    """
    Retorna: {"start": "frase_inicio", "middle": "frase_meio", "end": "frase_fim"}
    """
```

**Processo de Extração:**
1. Busca transcrições em português (manual tem prioridade sobre auto-gerada)
2. Divide a transcrição em segmentos temporais
3. Extrai a primeira frase completa de cada segmento
4. Limpa e formata o texto para análise PLN

**Tratamento de Erros:**
- Retorna frases vazias se `ENABLE_VIDEO_TRANSCRIPTION = False`
- Trata casos de vídeos sem transcrição disponível
- Gerencia erros de vídeos indisponíveis ou privados

### Utilização pelos Filtros PLN

Os filtros de PLN utilizam as frases extraídas para análise:

```python
# No filtro educacional, por exemplo
sentences = video_data.get('sentences', {})
combined_text = f"{video_data['title']} {video_data['description']} {sentences['start']} {sentences['middle']} {sentences['end']}"
```

**Filtros que Utilizam Transcrição:**
- **Análise de Sentimentos**: Analisa o tom emocional do conteúdo falado
- **Detecção de Toxicidade**: Identifica linguagem tóxica ou ofensiva
- **Classificação Educacional**: Avalia valor educacional baseado no conteúdo falado
- **Detecção de Linguagem Imprópria**: Detecta palavrões ou linguagem inadequada

### Configurações da YouTube API

As seguintes variáveis de configuração controlam o comportamento da integração:

#### `YOUTUBE_API_KEY`
- **Tipo**: String
- **Descrição**: Chave de API do Google para acessar a YouTube Data API v3
- **Obrigatório**: Sim
- **Configuração**: Definida via variável de ambiente ou arquivo `.env`

#### `MAX_SEARCH_RESULTS`
- **Tipo**: Integer
- **Padrão**: 8
- **Descrição**: Número máximo de vídeos retornados por busca
- **Impacto**: Afeta performance e custos da API

#### `ENABLE_VIDEO_TRANSCRIPTION`
- **Tipo**: Boolean  
- **Padrão**: False
- **Descrição**: Habilita/desabilita a obtenção de transcrições de vídeos
- **Impacto**: 
  - `True`: Filtros PLN funcionam com análise completa (título + descrição + transcrição)
  - `False`: Filtros PLN funcionam apenas com título e descrição

### Exemplo de Uso Completo

```python
from app.core.youtube import YouTubeAPI
from app.core.config import get_settings

settings = get_settings()
youtube_api = YouTubeAPI(settings.YOUTUBE_API_KEY)

# Busca com filtros PLN habilitados
nlp_filters = ["Análise de Sentimentos", "Detecção de Toxicidade"]
videos = await youtube_api.search_videos(
    query="desenhos educativos para crianças",
    max_results=10,
    video_duration="medium",
    nlp_filters_enabled=nlp_filters
)

# Cada vídeo retornado contém:
for video in videos:
    print(f"Título: {video['title']}")
    print(f"Duração: {video['duration_seconds']} segundos")
    print(f"Frases: {video['sentences']}")
```

## ⚙️ Variáveis de Configuração

O sistema utiliza três variáveis de configuração principais definidas em `app/core/config.py`:

### YOUTUBE_API_KEY
- **Descrição**: Chave de API do YouTube Data API v3
- **Tipo**: String
- **Obrigatório**: Sim
- **Função**: Permite acesso à API do YouTube para buscar informações de vídeos, incluindo metadados, estatísticas e detalhes dos canais
- **Como obter**: 
  1. Acesse o [Google Cloud Console](https://console.cloud.google.com/)
  2. Crie um projeto ou selecione um existente
  3. Ative a YouTube Data API v3
  4. Gere uma chave de API
- **Configuração**: Defina no arquivo `.env` como `YOUTUBE_API_KEY=sua_chave_aqui`

### MAX_SEARCH_RESULTS
- **Descrição**: Número máximo de vídeos retornados por busca
- **Tipo**: Integer
- **Padrão**: 50
- **Função**: Limita a quantidade de resultados processados por consulta, otimizando performance e custos de API
- **Recomendações**: 
  - Para testes: 10-20 vídeos
  - Para uso normal: 50 vídeos
  - Para análises extensas: até 100 vídeos (cuidado com limites de API)

### ENABLE_VIDEO_TRANSCRIPTION
- **Descrição**: Habilita ou desabilita a captura de transcrições de vídeos
- **Tipo**: Boolean
- **Padrão**: True
- **Função**: Controla se o sistema deve tentar obter transcrições automáticas ou manuais dos vídeos para análise de conteúdo
- **Impacto**: 
  - `True`: Permite análise completa de conteúdo textual dos vídeos
  - `False`: Análise baseada apenas em metadados (título, descrição, tags)
- **Considerações**: Desabilitar pode melhorar a performance, mas reduz a precisão dos filtros de PLN

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