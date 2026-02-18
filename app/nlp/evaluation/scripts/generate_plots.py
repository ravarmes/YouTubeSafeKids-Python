"""
Script para gerar gráficos e tabelas dos resultados do Grid Search.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load results
results_path = 'app/nlp/evaluation/results/toxicity_gridsearch/gridsearch_fast_20260206_202033.json'
with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results'][0]
fold_metrics = results['fold_metrics']

# Create output directory
output_dir = 'app/nlp/evaluation/results/plots'
os.makedirs(output_dir, exist_ok=True)

# Extract metrics
folds = [m['fold'] for m in fold_metrics]
f1_scores = [m['eval_f1'] for m in fold_metrics]
accuracies = [m['eval_accuracy'] for m in fold_metrics]
precisions = [m['eval_precision'] for m in fold_metrics]
recalls = [m['eval_recall'] for m in fold_metrics]
losses = [m['eval_loss'] for m in fold_metrics]

# 1. Bar chart - F1 Score e Accuracy por Fold
print("Gerando gráfico 1: Métricas por Fold...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(folds))
width = 0.35

bars1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='#2ecc71')
bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='#3498db')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Métricas por Fold - Cross-Validation (K=5)\nDetecção de Toxicidade', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {f}' for f in folds])
ax.legend(loc='lower right')
ax.set_ylim(0.7, 0.85)
ax.axhline(y=results['mean_f1'], color='#27ae60', linestyle='--', alpha=0.7, linewidth=2)
ax.axhline(y=results['mean_accuracy'], color='#2980b9', linestyle='--', alpha=0.7, linewidth=2)

# Add value labels on bars
for bar, val in zip(bars1, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/f1_por_fold.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Salvo: {output_dir}/f1_por_fold.png")

# 2. Box plot
print("Gerando gráfico 2: Box Plot...")
fig2, ax2 = plt.subplots(figsize=(8, 6))
data_box = [f1_scores, accuracies, precisions, recalls]
labels_box = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

bp = ax2.boxplot(data_box, labels=labels_box, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Distribuição das Métricas - 5-Fold CV\nDetecção de Toxicidade', fontsize=14, fontweight='bold')
ax2.set_ylim(0.7, 0.85)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/boxplot_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Salvo: {output_dir}/boxplot_metricas.png")

# 3. Summary table as image
print("Gerando gráfico 3: Tabela de Resultados...")
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.axis('off')

table_data = [
    ['Fold', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'Loss'],
]
for m in fold_metrics:
    table_data.append([
        str(m['fold']),
        f"{m['eval_f1']:.4f}",
        f"{m['eval_accuracy']:.4f}",
        f"{m['eval_precision']:.4f}",
        f"{m['eval_recall']:.4f}",
        f"{m['eval_loss']:.4f}"
    ])
table_data.append([
    'Média ± Std',
    f"{results['mean_f1']:.4f} ± {results['std_f1']:.4f}",
    f"{results['mean_accuracy']:.4f}",
    f"{np.mean(precisions):.4f}",
    f"{np.mean(recalls):.4f}",
    f"{results['mean_loss']:.4f}"
])

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style header row
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor('#3498db')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
    
# Style mean row
for j in range(len(table_data[0])):
    table[(len(table_data)-1, j)].set_facecolor('#ecf0f1')
    table[(len(table_data)-1, j)].set_text_props(fontweight='bold')

# Highlight best fold (fold 2)
for j in range(len(table_data[0])):
    table[(2, j)].set_facecolor('#d5f5e3')

plt.title('Resultados de Cross-Validation - Detecção de Toxicidade\n(Hiperparâmetros: epochs=3, batch=16, lr=5e-5, max_len=128)', 
          fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/tabela_resultados.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Salvo: {output_dir}/tabela_resultados.png")

# 4. Radar chart
print("Gerando gráfico 4: Radar Chart...")
fig4, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

categories = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
mean_values = [results['mean_f1'], results['mean_accuracy'], 
               np.mean(precisions), np.mean(recalls)]

# Add first value to close the circle
categories_plot = categories + [categories[0]]
values_plot = mean_values + [mean_values[0]]

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax4.plot(angles, values_plot, 'o-', linewidth=2, color='#3498db')
ax4.fill(angles, values_plot, alpha=0.25, color='#3498db')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=11)
ax4.set_ylim(0.7, 0.85)
ax4.set_title('Métricas Médias - Cross-Validation\nDetecção de Toxicidade', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/radar_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Salvo: {output_dir}/radar_metricas.png")

print("\n" + "="*50)
print("TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
print("="*50)
print(f"\nArquivos criados em: {output_dir}/")
print("  - f1_por_fold.png")
print("  - boxplot_metricas.png")
print("  - tabela_resultados.png")
print("  - radar_metricas.png")
