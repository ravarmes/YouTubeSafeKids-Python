"""
Script para gerar gráfico comparativo: Com vs Sem Pré-processamento.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = 'app/nlp/evaluation/results/plots'
os.makedirs(output_dir, exist_ok=True)

# Data
metrics = ['F1-Score', 'Accuracy']
raw_scores = [0.7893, 0.7902]      # Sem pré-processamento
pre_scores = [0.7127, 0.7119]      # Com pré-processamento (limpeza + lematização)

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, raw_scores, width, label='Sem Pré-processamento (Original)', color='#2ecc71')
rects2 = ax.bar(x + width/2, pre_scores, width, label='Com Bloqueio (Limpeza + Lematização)', color='#e74c3c')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Impacto do Pré-processamento no BERTimbau\nDetecção de Toxicidade', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0.6, 0.85)
ax.legend(loc='lower center')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
save_path = f'{output_dir}/comparacao_preprocessamento.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Gráfico salvo em: {save_path}")
