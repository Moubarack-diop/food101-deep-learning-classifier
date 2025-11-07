"""
Métriques d'évaluation pour Food-101
"""

import torch
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> list:
    """
    Calcule top-k accuracy

    Args:
        outputs: Logits du modèle (B, num_classes)
        labels: Labels ground truth (B,)
        topk: Tuple de k pour calculer top-k accuracy

    Returns:
        Liste des accuracies pour chaque k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        # Top-k prédictions
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        # Calculer l'accuracy pour chaque k
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """
    Évalue le modèle sur un dataset

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader
        device: Device
        topk: Top-k accuracies à calculer

    Returns:
        Dictionnaire des métriques
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_topk_accs = {k: [] for k in topk}

    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Top-k accuracies
            accs = calculate_accuracy(outputs, labels, topk=topk)
            for k, acc in zip(topk, accs):
                all_topk_accs[k].append(acc)

            # Prédictions top-1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculer les métriques moyennes
    metrics = {
        'loss': total_loss / len(dataloader),
    }

    # Top-k accuracies
    for k in topk:
        metrics[f'top{k}_acc'] = np.mean(all_topk_accs[k])

    # Métriques détaillées (precision, recall, f1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    metrics['precision'] = precision * 100
    metrics['recall'] = recall * 100
    metrics['f1_score'] = f1 * 100

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Affiche les métriques de manière formatée

    Args:
        metrics: Dictionnaire des métriques
        prefix: Préfixe pour l'affichage (ex: "Train", "Val")
    """
    print(f"\n{prefix} Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if 'acc' in key or 'precision' in key or 'recall' in key or 'f1' in key:
            print(f"  {key:15s}: {value:6.2f}%")
        else:
            print(f"  {key:15s}: {value:6.4f}")
    print("=" * 50)


def get_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Calcule la matrice de confusion

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader
        device: Device

    Returns:
        Matrice de confusion (num_classes, num_classes)
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


def get_classification_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = 'cuda'
) -> str:
    """
    Génère un rapport de classification détaillé

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader
        class_names: Liste des noms de classes
        device: Device

    Returns:
        Rapport de classification (string)
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=3
    )

    return report


class AverageMeter:
    """Calcule et stocke la moyenne et la valeur actuelle"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test des fonctions
    print("=== Test des métriques ===\n")

    # Simuler des prédictions
    batch_size = 32
    num_classes = 101

    outputs = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test accuracy
    top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\nAverageMeter test: avg={meter.avg:.2f}, sum={meter.sum}")

    print("\n Tests réussis!")
