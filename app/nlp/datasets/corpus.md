# Guia de Interpretação dos Filtros do Modelo

Este documento descreve o significado dos valores retornados pelos modelos ajustados para cada filtro:  
- **Análise de Sentimentos (AS)**  
- **Toxicidade (TOX)**  
- **Linguagem Imprópria (LI)**  
- **Tópicos Educacionais (TE)**  

Cada modelo retorna um **vetor de probabilidades**, e o índice com maior valor corresponde à classe prevista.

---

## 1. Análise de Sentimentos (AS)

- **0 - Negativo**  
  Transmite tristeza, raiva, frustração, medo ou crítica.  
  *Exemplo:* `"Você nunca faz nada direito!"`

- **1 - Neutro**  
  Informação ou fala sem carga emocional relevante.  
  *Exemplo:* `"O cachorro correu pelo quintal."`

- **2 - Positivo**  
  Expressa alegria, empolgação, carinho ou elogio.  
  *Exemplo:* `"Você é o melhor amigo que eu poderia ter!"`

---

## 2. Toxicidade (TOX)

- **0 - Não tóxico**  
  Linguagem amigável e respeitosa.  
  *Exemplo:* `"Vamos brincar juntos?"`

- **1 - Leve**  
  Pequenas provocações ou ironias que não são extremamente ofensivas.  
  *Exemplo:* `"Você é meio atrapalhado, hein?"`

- **2 - Moderado**  
  Ofensas mais diretas ou hostilidade clara, mas não extremamente graves.  
  *Exemplo:* `"Você é muito burro!"`

- **3 - Severa**  
  Discurso de ódio, humilhação extrema ou incitação de violência.  
  *Exemplo:* `"Ninguém gosta de você, desapareça daqui!"`

---

## 3. Linguagem Imprópria (LI)

- **0 - Nenhuma**  
  Sem palavras inadequadas.  
  *Exemplo:* `"Olha que desenho bonito!"`

- **1 - Leve**  
  Expressões inapropriadas, mas não extremamente ofensivas.  
  *Exemplo:* `"Isso é uma droga de brinquedo."`

- **2 - Severa**  
  Palavrões ou xingamentos explícitos.  
  *Exemplo:* `"Vai se ferrar, seu idiota!"`

---

## 4. Tópicos Educacionais (TE)

- **0 - Não educacional**  
  Apenas entretenimento, sem conteúdo instrutivo.  
  *Exemplo:* `"Vamos ver quem ganha essa corrida!"`

- **1 - Parcialmente educacional**  
  Mistura de entretenimento com alguma informação útil.  
  *Exemplo:* `"O dinossauro vivia há milhões de anos... Agora vamos brincar!"`

- **2 - Educacional**  
  Conteúdo totalmente voltado para ensinar algo.  
  *Exemplo:* `"Hoje vamos aprender como somar números até 10."`

---

## Como interpretar a saída

Exemplo de saída do modelo para **Análise de Sentimentos (AS):**

