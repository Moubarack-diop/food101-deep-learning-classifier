"""
Script d'évaluation du modèle Food-101
"""

import sys
import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.resnet_classifier import load_checkpoint
from src.data.dataset import get_dataloaders, Food101Dataset
from src.data.transforms import get_transforms
from src.utils.metrics import (
    evaluate_model,
    print_metrics,
    get_confusion_matrix,
    get_classification_report
)
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_top_errors,
    plot_class_accuracies
)
from src.training.config import Config


def evaluate_checkpoint(checkpoint_path, config=Config, save_results=True):
    """
    Évalue un modèle depuis un checkpoint

    Args:
        checkpoint_path: Chemin vers le checkpoint .pth
        config: Configuration
        save_results: Sauvegarder les résultats et visualisations

    Returns:
        metrics: Dictionnaire des métriques
    """
    print("\n" + "="*60)
    print("ÉVALUATION DU MODÈLE FOOD-101")
    print("="*60)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    # Charger le modèle
    print(f"\n Chargement du modèle depuis {checkpoint_path}...")
    model = load_checkpoint(checkpoint_path, num_classes=config.NUM_CLASSES, device=device)

    # Charger les données de test
    print("\n Chargement des données de test...")
    _, test_transform = get_transforms(img_size=config.IMG_SIZE)

    test_dataset = Food101Dataset(
        root_dir=str(config.DATA_DIR),
        split='test',
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Évaluation
    print("\n Évaluation en cours...")
    metrics = evaluate_model(model, test_loader, device=device, topk=(1, 5))

    # Afficher les résultats
    print_metrics(metrics, prefix="Test Set")

    # Matrice de confusion
    print("\n Calcul de la matrice de confusion...")
    cm = get_confusion_matrix(model, test_loader, device=device)

    if save_results:
        results_dir = config.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les métriques
        import json
        metrics_path = results_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Métriques sauvegardées: {metrics_path}")

        # Sauvegarder la matrice de confusion
        cm_path = results_dir / 'confusion_matrix.npy'
        np.save(cm_path, cm)
        print(f"✓ Matrice de confusion sauvegardée: {cm_path}")

        # Visualisations
        print("\n Génération des visualisations...")

        # 1. Matrice de confusion (top 20 classes confondues)
        plot_confusion_matrix(
            cm,
            class_names=test_dataset.classes,
            top_n=20,
            save_path=str(results_dir / 'confusion_matrix_top20.png')
        )

        # 2. Paires les plus confondues
        plot_top_errors(
            cm,
            class_names=test_dataset.classes,
            top_n=15,
            save_path=str(results_dir / 'top_errors.png')
        )

        # 3. Meilleures classes
        plot_class_accuracies(
            cm,
            class_names=test_dataset.classes,
            top_n=20,
            mode='best',
            save_path=str(results_dir / 'best_classes.png')
        )

        # 4. Pires classes
        plot_class_accuracies(
            cm,
            class_names=test_dataset.classes,
            top_n=20,
            mode='worst',
            save_path=str(results_dir / 'worst_classes.png')
        )

        # Rapport de classification détaillé
        print("\n Génération du rapport de classification...")
        report = get_classification_report(
            model,
            test_loader,
            class_names=test_dataset.classes,
            device=device
        )

        report_path = results_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Rapport sauvegardé: {report_path}")

        print("\n" + "="*60)
        print(" ÉVALUATION TERMINÉE")
        print("="*60)
        print(f"\n Résultats disponibles dans: {results_dir}")

    return metrics, cm


def compare_with_baseline():
    """Compare les résultats avec l'article original (2014)"""
    print("\n" + "="*60)
    print("COMPARAISON AVEC L'ARTICLE ORIGINAL (2014)")
    print("="*60)

    baseline = {
        'method': 'Random Forests (2014)',
        'top1_acc': 50.76,
        'top5_acc': 80.0,
        'inference_time': 500  # ms
    }

    # Charger nos résultats
    results_path = Config.RESULTS_DIR / 'test_metrics.json'
    if results_path.exists():
        import json
        with open(results_path, 'r') as f:
            our_metrics = json.load(f)

        print("\n Comparaison des performances:\n")
        print(f"{'Métrique':<20} {'Article 2014':<15} {'Notre Modèle':<15} {'Gain':<15}")
        print("-" * 65)

        # Top-1 Accuracy
        our_top1 = our_metrics.get('top1_acc', 0)
        gain_top1 = our_top1 - baseline['top1_acc']
        print(f"{'Top-1 Accuracy':<20} {baseline['top1_acc']:>13.2f}% {our_top1:>13.2f}% {gain_top1:>+13.2f}%")

        # Top-5 Accuracy
        our_top5 = our_metrics.get('top5_acc', 0)
        gain_top5 = our_top5 - baseline['top5_acc']
        print(f"{'Top-5 Accuracy':<20} {baseline['top5_acc']:>13.2f}% {our_top5:>13.2f}% {gain_top5:>+13.2f}%")

        # F1-Score
        our_f1 = our_metrics.get('f1_score', 0)
        baseline_f1 = 50.0  # Estimation
        gain_f1 = our_f1 - baseline_f1
        print(f"{'F1-Score':<20} {baseline_f1:>13.2f}% {our_f1:>13.2f}% {gain_f1:>+13.2f}%")

        print("\n" + "="*60)

        # Analyse
        if our_top1 >= 87:
            print(" OBJECTIF ATTEINT! (87-90% top-1 accuracy)")
        elif our_top1 >= 85:
            print(" Bon résultat (85-87% top-1 accuracy)")
        else:
            print("  Objectif non atteint (<85% top-1 accuracy)")

        print("="*60)

    else:
        print("\n Résultats non trouvés. Veuillez d'abord évaluer le modèle.")


def main():
    parser = argparse.ArgumentParser(description="Évaluer le modèle Food-101")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Chemin vers le checkpoint'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Ne pas sauvegarder les résultats'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Comparer avec l\'article original'
    )

    args = parser.parse_args()

    # Évaluation
    if Path(args.checkpoint).exists():
        metrics, cm = evaluate_checkpoint(
            args.checkpoint,
            config=Config,
            save_results=not args.no_save
        )

        # Comparaison optionnelle
        if args.compare:
            compare_with_baseline()
    else:
        print(f"\n Checkpoint non trouvé: {args.checkpoint}")
        print("Veuillez d'abord entraîner le modèle avec trainer.py")


if __name__ == "__main__":
    main()
