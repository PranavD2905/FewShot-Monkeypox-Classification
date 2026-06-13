"""
Create a grid of confusion matrices for visual comparison
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Confusion matrix files
proto_cms = [
    ('outs/proto_resnet50/confusion.png', 'Prototypical - ResNet50'),
    ('outs/proto_efficientnet_b0/confusion.png', 'Prototypical - EfficientNet-B0'),
    ('outs/proto_densenet121/confusion.png', 'Prototypical - DenseNet121'),
    ('outs/proto_mobilenet_v3/confusion.png', 'Prototypical - MobileNetV3'),
]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (cm_file, title) in enumerate(proto_cms):
    if Path(cm_file).exists():
        img = mpimg.imread(cm_file)
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, f'{title}\n(Not Found)', 
                      ha='center', va='center', fontsize=12)
        axes[idx].axis('off')

plt.suptitle('Confusion Matrices - Prototypical Networks (500 Episodes)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('confusion_matrices_prototypical.png', dpi=300, bbox_inches='tight')
print("Saved confusion_matrices_prototypical.png")
plt.close()

print("\nGenerated Files Summary:")
print("=" * 70)
print("1. training_curves_prototypical.png - Learning curves for all models")
print("2. confusion_matrices_prototypical.png - Confusion matrix grid")
print("3. detailed_metrics.csv - Comprehensive metrics comparison")
print("4. Individual confusion matrices in outs/ subdirectories")
print("=" * 70)
