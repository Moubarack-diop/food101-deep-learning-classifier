"""
Transformations d'images pour Food-101
"""

from typing import Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Constantes ImageNet pour normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Taille des images
IMG_SIZE = 224


def get_train_transforms(
    img_size: int = IMG_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD,
    augmentation_level: str = 'medium'
) -> transforms.Compose:
    """
    Transformations pour l'entraînement avec data augmentation

    Args:
        img_size: Taille finale des images
        mean: Moyenne pour normalisation
        std: Écart-type pour normalisation
        augmentation_level: 'light', 'medium', ou 'heavy'

    Returns:
        Composition de transformations
    """

    if augmentation_level == 'light':
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif augmentation_level == 'medium':
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
        ])

    elif augmentation_level == 'heavy':
        # OPTIMISÉ: Augmentation plus agressive pour meilleure généralisation
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Crop avec scale variable
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),  # Augmenté de 15 à 20 degrés
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Plus agressif
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # Plus de variation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.25))  # Probabilité et taille augmentées
        ])

    else:
        raise ValueError(f"augmentation_level doit être 'light', 'medium' ou 'heavy', pas '{augmentation_level}'")


def get_test_transforms(
    img_size: int = IMG_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> transforms.Compose:
    """
    Transformations pour l'évaluation (sans augmentation)

    Args:
        img_size: Taille finale des images
        mean: Moyenne pour normalisation
        std: Écart-type pour normalisation

    Returns:
        Composition de transformations
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_transforms(
    img_size: int = IMG_SIZE,
    augmentation_level: str = 'medium'
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Retourne les transformations pour train et test

    Args:
        img_size: Taille des images
        augmentation_level: Niveau d'augmentation

    Returns:
        (train_transform, test_transform)
    """
    train_transform = get_train_transforms(img_size=img_size, augmentation_level=augmentation_level)
    test_transform = get_test_transforms(img_size=img_size)

    return train_transform, test_transform


def denormalize(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Dénormalise un tensor pour visualisation

    Args:
        tensor: Tensor normalisé (C, H, W) ou (B, C, H, W)
        mean: Moyenne utilisée pour normalisation
        std: Écart-type utilisé pour normalisation

    Returns:
        Tensor dénormalisé
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def tensor_to_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convertit un tensor normalisé en image visualisable

    Args:
        tensor: Tensor (C, H, W) ou (B, C, H, W)

    Returns:
        Tensor dans [0, 1]
    """
    denorm = denormalize(tensor)
    return torch.clamp(denorm, 0, 1)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Applique MixUp augmentation

    MixUp crée des exemples d'entraînement virtuels en mélangeant deux images et leurs labels.
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)

    Args:
        x: Batch d'images (B, C, H, W)
        y: Labels (B,)
        alpha: Paramètre de la distribution Beta

    Returns:
        mixed_x: Images mélangées
        y_a: Labels originaux
        y_b: Labels mélangés
        lam: Lambda (coefficient de mélange)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Applique CutMix augmentation

    CutMix coupe et colle des patchs d'une image dans une autre et mélange les labels proportionnellement.
    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)

    Args:
        x: Batch d'images (B, C, H, W)
        y: Labels (B,)
        alpha: Paramètre de la distribution Beta

    Returns:
        mixed_x: Images avec patchs coupés/collés
        y_a: Labels originaux
        y_b: Labels mélangés
        lam: Lambda (proportion de l'image originale)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Obtenir les dimensions
    _, _, H, W = x.size()

    # Calculer les coordonnées de la boîte à couper
    cut_rat = torch.sqrt(torch.tensor(1.0 - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Centre de la boîte (uniforme)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    # Coordonnées de la boîte
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)

    # Appliquer CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Ajuster lambda pour refléter la proportion réelle
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calcule la loss pour MixUp/CutMix

    Args:
        criterion: Fonction de loss (ex: CrossEntropyLoss)
        pred: Prédictions du modèle
        y_a: Labels originaux
        y_b: Labels mélangés
        lam: Coefficient de mélange

    Returns:
        Loss mixée
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    # Test des transformations
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    # Créer une image de test
    dummy_img = Image.new('RGB', (500, 400), color=(73, 109, 137))

    # Tester les transformations
    train_transform, test_transform = get_transforms(augmentation_level='medium')

    # Appliquer
    train_img = train_transform(dummy_img)
    test_img = test_transform(dummy_img)

    print(f"Train image shape: {train_img.shape}")
    print(f"Test image shape: {test_img.shape}")

    # Dénormaliser pour visualisation
    train_img_vis = tensor_to_image(train_img)
    test_img_vis = tensor_to_image(test_img)

    print(f"Train min/max: {train_img_vis.min():.3f}/{train_img_vis.max():.3f}")
    print(f"Test min/max: {test_img_vis.min():.3f}/{test_img_vis.max():.3f}")

    print("\n✓ Transformations testées avec succès")
