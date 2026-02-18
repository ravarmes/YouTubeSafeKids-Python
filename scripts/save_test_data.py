import sys
import os
import pandas as pd
from pathlib import Path

# Adiciona diretório raiz
sys.path.append(str(Path(__file__).parent.parent))

from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test

# 1. Pega a mesma divisão usada no treino (graças ao random_state=42)
print("Recriando a divisão dos dados...")
_, (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)

# 2. Cria um DataFrame com esses dados
df_test = pd.DataFrame({
    'text': X_test,
    'label': y_test
})

# 3. Salva em CSV
output_path = 'data/test_dataset_20percent.csv'
df_test.to_csv(output_path, index=False)

print(f"✅ Arquivo salvo com sucesso em: {output_path}")
print(f"Quantidade de amostras: {len(df_test)}")