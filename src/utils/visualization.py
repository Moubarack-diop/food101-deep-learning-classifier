"""
Fonctions de visualisation pour Food-101
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path


def plot_training_curves(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Affiche les courbes d'entraînement

    Args:
        train_losses: Liste des losses d'entraînement
        train_accs: Liste des accuracies d'entraînement
        val_losses: Liste des losses de validation
        val_accs: Liste des accuracies de validation
        save_path: Chemin pour sauvegarder la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 18),
    top_n: Optional[int] = None
):
    """
    Affiche la matrice de confusion

    Args:
        cm: Matrice de confusion (num_classes, num_classes)
        class_names: Liste des noms de classes
        save_path: Chemin pour sauvegarder la figure
        figsize: Taille de la figure
        top_n: Afficher seulement les top N classes les plus confondues
    """
    if top_n:
        # Sélectionner les classes les plus confondues
        errors = cm.sum(axis=1) - np.diag(cm)
        top_indices = np.argsort(errors)[-top_n:]
        cm = cm[top_indices][:, top_indices]
        if class_names:
            class_names = [class_names[i] for i in top_indices]

    # Normaliser par ligne (recall)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Proportion'},
        xticklabels=class_names if class_names else False,
        yticklabels=class_names if class_names else False
    )

    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')

    if class_names:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_top_errors(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = None
):
    """
    Affiche les paires de classes les plus confondues

    Args:
        cm: Matrice de confusion
        class_names: Liste des noms de classes
        top_n: Nombre de paires à afficher
        save_path: Chemin pour sauvegarder
    """
    # Extraire les erreurs (hors diagonale)
    errors = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j:
                errors.append((cm[i, j], class_names[i], class_names[j]))

    # Trier par nombre d'erreurs
    errors.sort(reverse=True)
    top_errors = errors[:top_n]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))

    counts = [err[0] for err in top_errors]
    labels = [f"{err[1]}\n→ {err[2]}" for err in top_errors]

    bars = ax.barh(range(len(counts)), counts, color='coral')
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title(f'Top {top_n} Most Confused Class Pairs', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(width)}', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_class_accuracies(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 20,
    mode: str = 'best',
    save_path: Optional[str] = None
):
    """
    Affiche les meilleures ou pires classes

    Args:
        cm: Matrice de confusion
        class_names: Liste des noms de classes
        top_n: Nombre de classes à afficher
        mode: 'best' ou 'worst'
        save_path: Chemin pour sauvegarder
    """
    # Calculer l'accuracy par classe
    accuracies = []
    for i in range(len(cm)):
        acc = cm[i, i] / cm[i].sum() * 100
        accuracies.append((acc, class_names[i]))

    # Trier
    reverse = (mode == 'best')
    accuracies.sort(reverse=reverse)
    top_accs = accuracies[:top_n]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 10))

    accs = [acc[0] for acc in top_accs]
    labels = [acc[1] for acc in top_accs]

    colors = ['green' if mode == 'best' else 'red'] * len(accs)
    bars = ax.barh(range(len(accs)), accs, color=colors, alpha=0.7)

    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    title = f'Top {top_n} {"Best" if mode == "best" else "Worst"} Performing Classes'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 100)

    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    num_images: int = 16,
    denormalize_fn: Optional[callable] = None,
    save_path: Optional[str] = None
):
    """
    Visualise des prédictions du modèle

    Args:
        images: Batch d'images (B, C, H, W)
        labels: Labels ground truth (B,)
        predictions: Prédictions du modèle (B, num_classes)
        class_names: Liste des noms de classes
        num_images: Nombre d'images à afficher
        denormalize_fn: Fonction pour dénormaliser les images
        save_path: Chemin pour sauvegarder
    """
    # Convertir en numpy
    images = images[:num_images].cpu()
    labels = labels[:num_images].cpu().numpy()

    # Prédictions top-1
    _, pred_indices = torch.max(predictions[:num_images], 1)
    pred_indices = pred_indices.cpu().numpy()

    # Dénormaliser si nécessaire
    if denormalize_fn:
        images = denormalize_fn(images)

    # Créer la grille
    n_cols = 4
    n_rows = (num_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]

        # Image
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)

        # Titre avec true/pred labels
        true_label = class_names[labels[i]]
        pred_label = class_names[pred_indices[i]]

        color = 'green' if labels[i] == pred_indices[i] else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')

    # Masquer les axes vides
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("=== Test des fonctions de visualisation ===\n")

    # Test training curves
    train_losses = [2.5, 2.0, 1.5, 1.2, 1.0]
    train_accs = [30, 50, 65, 75, 80]
    val_losses = [2.6, 2.1, 1.6, 1.3, 1.1]
    val_accs = [28, 48, 63, 73, 78]

    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    print("\n Visualisations testées!")
