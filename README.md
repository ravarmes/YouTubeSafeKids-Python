# YouTube Safe Kids

<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/YouTubeSafeKids-Python/blob/main/app/static/img/logo.jpg" />
    <img alt="YouTubeSafeKids" src="https://github.com/ravarmes/YouTubeSafeKids-Python/blob/main/app/static/img/logo.svg" />
</h1>

<h3 align="center">
  Plataforma de Pesquisa de Conteúdos Infantis Seguros no YouTube
</h3>

<p align="center">Filtragem Avançada de Conteúdos Infantis</p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/YouTubeSafeKids-Python?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/YouTubeSafeKids-Python/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/YouTubeSafeKids-Python?style=social">
  </a>
</p>

## Sobre o projeto

Uma aplicação web que ajuda a encontrar vídeos seguros e apropriados para crianças no YouTube.

## Funcionalidades

- Busca de vídeos no YouTube Kids
- Filtros de conteúdo:
  - Duração
  - Faixa etária
  - Conteúdo educacional
  - Toxicidade
  - Linguagem
  - Diversidade
  - Interatividade
  - Engajamento
  - Sentimento
  - Conteúdo sensível
  
## :notebook_with_decorative_cover: Arquitetura do Sistema <a name="-architecture"/></a>

<img alt="YouTubeSafeKids-Metodologia" src="https://github.com/ravarmes/YouTubeSafeKids-Python/blob/main/app/static/img/YouTubeSafeKids-Metodologia.png" />

## :notebook_with_decorative_cover: Protótipo da Plataforma <a name="-architecture"/></a>

<img alt="YouTubeSafeKids-Prototipo" src="https://github.com/ravarmes/YouTubeSafeKids-Python/blob/main/app/static/img/YouTubeSafeKids-Prototipo.png" />


## Requisitos

- Python 3.8+
- Chave de API do YouTube
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/YouTubeSafeKids-Python.git
cd YouTubeSafeKids-Python
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## Uso

1. Inicie o servidor:
```bash
uvicorn app.main:app --reload --log-level debug
```

2. Acesse a aplicação em `http://localhost:8000`

3. Use a barra de busca para encontrar vídeos

4. Ajuste os filtros conforme necessário

## Estrutura do Projeto

```
app/
├── api/
│   ├── endpoints/
│   │   └── videos.py
│   └── dependencies.py
├── core/
│   ├── config.py
│   ├── logging.py
│   └── youtube.py
├── filters/
│   ├── base.py
│   ├── duration.py
│   ├── age_rating.py
│   └── ... (outros filtros)
├── static/
│   ├── css/
│   └── js/
├── templates/
└── main.py
```

## Logs

Os logs são salvos em:
- Console: Logs em tempo real
- Arquivo: `logs/app.log`

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Desenvolvimento

O projeto está em desenvolvimento ativo, com atualizações frequentes incluindo:
- Novas implementações de filtros
- Treinamento de modelos
- Otimizações de interface

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## :email: Contact

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**