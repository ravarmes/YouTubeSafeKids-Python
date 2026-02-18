# Guia de Preparação e Execução no Servidor Linux

Este guia descreve passo-a-passo como migrar o projeto para o servidor, configurar o ambiente e executar o treinamento.

## 1. O que Copiar para o Servidor

Você não precisa copiar tudo (o projeto tem 19GB por causa de modelos e ambeinetes antigos). **Copie apenas os itens essenciais listados abaixo:**

### Pastas e Arquivos Essenciais
*   📂 `app/` (Todo o código fonte)
*   📂 `data/` (Contém `corpus_toxicidade.csv` - **Indispensável**)
*   📄 `requirements.txt`
*   📄 `tutorial_full_grid_search.ipynb` (Notebook de execução)
*   📄 `README.md` (Opcional)

### ❌ O QUE NÃO COPIAR
*   🚫 `venv_nlp/` (Ambiente virtual do Windows - não funciona no Linux)
*   🚫 `models/` (Modelos pesados. O script baixará o Bertimbau automaticamente. Se tiver backups antigos, só copie se estritamente necessário)
*   🚫 `__pycache__/` (Arquivos temporários do Python)
*   🚫 `evaluation_results/` (Resultados antigos, opcional)

---

## 2. Configuração do Ambiente no Servidor (Terminal)

No terminal do servidor (acess via SSH ou terminal do Jupyter), execute os seguintes comandos na pasta onde você colocou os arquivos:

1.  **Criação do Ambiente Virtual**
    ```bash
    # Se não tiver o módulo venv: sudo apt install python3-venv
    python3 -m venv venv
    ```

2.  **Ativação do Ambiente**
    ```bash
    source venv/bin/activate
    ```

3.  **Instalação do PyTorch (Com suporte a GPU)**
    *Recomendamos instalar o PyTorch manualmente primeiro para garantir o suporte a CUDA (Nvidia).*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *(Nota: Se o servidor tiver CUDA 12, use `cu121` no lugar de `cu118`)*
45→    **Sem GPU (CPU apenas):**
46→    ```bash
47→    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
48→    ```

4.  **Instalação das Dependências do Projeto**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Instalação do Kernel para Jupyter**
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=venv_nlp --display-name "Python (Brasnam NLP)"
    ```

---

## 3. Execução no Jupyter

1.  Abra o Jupyter no servidor.
2.  Navegue até a pasta do projeto e abra o arquivo:
    👉 **`tutorial_full_grid_search.ipynb`**
3.  No menu do notebook, vá em **Kernel** -> **Change Kernel** e selecione **"Python (Brasnam NLP)"** (ou o nome que apareceu no passo 2.5).
4.  Execute as células sequencialmente.

### Dica: Execução em Background (Opcional)
Se preferir rodar via terminal para não depender da conexão do navegador aberta:

```bash
source venv/bin/activate
# nohup mantém rodando mesmo se sair do SSH
nohup python -m app.nlp.training.train_toxicity_gridsearch --mode full > training_log.txt 2>&1 &
```
75→*Sem pré-processamento:* não use a flag `--preprocess` (o comando acima já está SEM).
76→
77→### Modos de Execução
78→- Validação rápida (menos custosa): 
79→```bash
80→nohup python -m app.nlp.training.train_toxicity_gridsearch --mode fast > training_log.txt 2>&1 &
81→```
82→- Experimento completo (artigo): 
83→```bash
84→nohup python -m app.nlp.training.train_toxicity_gridsearch --mode full > training_log.txt 2>&1 &
85→```
86→
87→### Como acompanhar o treinamento (Debian)
88→- Logs em tempo real:
89→```bash
90→tail -f training_log.txt
91→```
92→- Atualização em “janela”:
93→```bash
94→watch -n 5 'tail -n 50 training_log.txt'
95→```
96→- Filtrar métricas:
97→```bash
98→grep -E 'eval_|loss|epoch' -n training_log.txt
99→```
100→- Se tiver GPU Nvidia:
101→```bash
102→watch -n 5 nvidia-smi
103→```
104→
105→### Saída e modelos gerados
106→- Apenas o modelo final (melhor configuração) é salvo em: `app/nlp/models/trained/TOX_*`
107→- Resultados do Grid Search: `app/nlp/evaluation/results/toxicity_gridsearch/gridsearch_<mode>_raw_<timestamp>.json`
108→- Para avaliar no holdout (20%), após o treino:
109→```bash
110→python -m app.nlp.evaluation.evaluate_toxicity
111→# Se tiver treinado com pré-processamento (não é o caso aqui):
112→# python -m app.nlp.evaluation.evaluate_toxicity --preprocess
113→```
