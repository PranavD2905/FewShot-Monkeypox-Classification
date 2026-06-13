"""
Create comparison visualizations for prototypical networks vs classifiers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the detailed metrics
df = pd.read_csv('detailed_metrics.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 10))

# Color palette
colors_proto = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
colors_class = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b']

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
proto_df = df[df['model_type'] == 'prototypical'].sort_values('accuracy', ascending=False)
class_df = df[df['model_type'] == 'classifier'].sort_values('accuracy', ascending=False)

x = np.arange(len(proto_df))
width = 0.35

bars1 = ax1.bar(x - width/2, proto_df['accuracy'], width, label='Prototypical', 
                color=colors_proto, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, class_df['accuracy'], width, label='Baseline Model',
                color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Backbone', fontweight='bold')
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['EfficientNet', 'MobileNet', 'DenseNet', 'ResNet'], rotation=45, ha='right')
ax1.legend()
ax1.set_ylim([0.75, 1.0])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 3, 2)
bars1 = ax2.bar(x - width/2, proto_df['f1'], width, label='Prototypical',
                color=colors_proto, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, class_df['f1'], width, label='Baseline Model',
                color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Backbone', fontweight='bold')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['EfficientNet', 'MobileNet', 'DenseNet', 'ResNet'], rotation=45, ha='right')
ax2.legend()
ax2.set_ylim([0.75, 1.0])
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 3. AUC Comparison
ax3 = plt.subplot(2, 3, 3)
bars1 = ax3.bar(x - width/2, proto_df['auc'], width, label='Prototypical',
                color=colors_proto, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, class_df['auc'], width, label='Baseline Model',
                color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Backbone', fontweight='bold')
ax3.set_ylabel('AUC', fontweight='bold')
ax3.set_title('AUC Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['EfficientNet', 'MobileNet', 'DenseNet', 'ResNet'], rotation=45, ha='right')
ax3.legend()
ax3.set_ylim([0.95, 1.01])
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 4. Precision-Recall Trade-off
ax4 = plt.subplot(2, 3, 4)
for idx, row in proto_df.iterrows():
    ax4.scatter(row['recall'], row['precision'], s=200, 
               label=f"Proto-{row['backbone'].replace('_', ' ').title()}", 
               alpha=0.7, edgecolors='black', linewidth=2, marker='o')

for idx, row in class_df.iterrows():
    ax4.scatter(row['recall'], row['precision'], s=200,
               label=f"Base-{row['backbone'].replace('_', ' ').title()}",
               alpha=0.7, edgecolors='black', linewidth=2, marker='s')

ax4.set_xlabel('Recall', fontweight='bold')
ax4.set_ylabel('Precision', fontweight='bold')
ax4.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax4.legend(fontsize=8, loc='lower left')
ax4.grid(alpha=0.3)
ax4.set_xlim([0.6, 1.0])
ax4.set_ylim([0.9, 1.01])

# 5. Performance Improvement
ax5 = plt.subplot(2, 3, 5)
improvements = []
backbones_sorted = []
for backbone in proto_df['backbone'].values:
    proto_acc = proto_df[proto_df['backbone'] == backbone]['accuracy'].values[0]
    class_acc = class_df[class_df['backbone'] == backbone]['accuracy'].values[0]
    improvement = (proto_acc - class_acc) * 100
    improvements.append(improvement)
    backbones_sorted.append(backbone.replace('_', ' ').title())

bars = ax5.barh(backbones_sorted, improvements, color=colors_proto, 
                alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Accuracy Improvement (%)', fontweight='bold')
ax5.set_title('Prototypical Network Advantage', fontsize=14, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax5.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# 6. Metrics Heatmap
ax6 = plt.subplot(2, 3, 6)
heatmap_data = proto_df[['backbone', 'accuracy', 'precision', 'recall', 'f1', 'auc']].copy()
heatmap_data['backbone'] = heatmap_data['backbone'].apply(lambda x: x.replace('_', ' ').title())
heatmap_data = heatmap_data.set_index('backbone')

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.85, vmax=1.0, ax=ax6, cbar_kws={'label': 'Score'},
            linewidths=1, linecolor='black')
ax6.set_title('Prototypical Network Metrics Heatmap', fontsize=14, fontweight='bold')
ax6.set_ylabel('')

plt.suptitle('Few-Shot Learning: Prototypical Networks vs Standard Classifiers', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved model_comparison.png")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nAverage Prototypical Accuracy: {proto_df['accuracy'].mean():.4f}")
print(f"Average Classifier Accuracy:    {class_df['accuracy'].mean():.4f}")
print(f"Average Improvement:            {np.mean(improvements):.2f}%")
print(f"\nBest Model: {proto_df.iloc[0]['backbone'].replace('_', ' ').title()}")
print(f"  - Accuracy:  {proto_df.iloc[0]['accuracy']:.4f}")
print(f"  - F1-Score:  {proto_df.iloc[0]['f1']:.4f}")
print(f"  - AUC:       {proto_df.iloc[0]['auc']:.4f}")
print("="*70)
