import matplotlib.pyplot as plt
import numpy as np

def generate_ieee_graphs():
    """
    Generates sample accuracy and loss graphs suitable for an IEEE conference paper.
    If you have real training data (e.g., from train.py), replace the arrays here.
    """
    epochs = np.arange(1, 11)
    
    # Sample training/validation data for ViT
    vit_train_acc = np.array([0.65, 0.75, 0.82, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98])
    vit_val_acc = np.array([0.60, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.94, 0.95])
    
    # Sample training/validation data for Swin Transformer
    swin_train_acc = np.array([0.62, 0.74, 0.80, 0.86, 0.89, 0.92, 0.94, 0.95, 0.96, 0.97])
    swin_val_acc = np.array([0.58, 0.70, 0.78, 0.84, 0.87, 0.89, 0.90, 0.91, 0.93, 0.94])
    
    plt.figure(figsize=(10, 6))
    
    # Plotting styles to make it IEEE standard (black/white/gray or distinct markers)
    plt.plot(epochs, vit_train_acc, marker='o', label='ViT Training Acc', linestyle='-', color='#1f77b4')
    plt.plot(epochs, vit_val_acc, marker='s', label='ViT Validation Acc', linestyle='--', color='#1f77b4')
    
    plt.plot(epochs, swin_train_acc, marker='^', label='Swin Training Acc', linestyle='-', color='#ff7f0e')
    plt.plot(epochs, swin_val_acc, marker='d', label='Swin Validation Acc', linestyle='--', color='#ff7f0e')

    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Accuracy Convergence', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save in high resolution suitable for publishing
    plt.savefig('accuracy_comparison_ieee.png', dpi=300, bbox_inches='tight')
    print("Saved IEEE standard accuracy graph as 'accuracy_comparison_ieee.png'")

if __name__ == '__main__':
    generate_ieee_graphs()
