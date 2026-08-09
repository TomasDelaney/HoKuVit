import matplotlib.pyplot as plt
import numpy as np


def plot_training(train_losses, valid_losses, valid_accuracies, model_name):
    """Save a loss + validation-accuracy figure to plots/{model_name}_training.png."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
    if valid_losses:
        val_epochs = np.linspace(0, len(train_losses) - 1, len(valid_losses))
        axes[0].plot(val_epochs, valid_losses, label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if valid_accuracies:
        val_epochs = np.linspace(0, len(train_losses) - 1, len(valid_accuracies))
        axes[1].plot(val_epochs, valid_accuracies, label='Val Accuracy', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_debug_data(model_name, all_grad_norms):
    """Save a gradient-norm-over-time figure, if any were recorded."""
    if not all_grad_norms:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(all_grad_norms, alpha=0.7)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms During Training')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_grad_norms.png', dpi=300, bbox_inches='tight')
    plt.close()
