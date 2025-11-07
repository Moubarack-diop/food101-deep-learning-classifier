"""
Dataset Food-101 pour PyTorch
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image


class Food101Dataset(Dataset):
    """
    Dataset Food-101

    Args:
        root_dir: Chemin vers le dossier food-101/
        split: 'train' ou 'test'
        transform: Transformations à appliquer aux images
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        assert split in ['train', 'test'], "split doit être 'train' ou 'test'"

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Chemins
        self.images_dir = self.root_dir / "images"
        self.meta_dir = self.root_dir / "meta"

        # Charger les classes
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Charger les chemins des images
        self.image_paths, self.labels = self._load_split()

        print(f"✓ Food101Dataset ({split}): {len(self)} images, {len(self.classes)} classes")

    def _load_classes(self) -> List[str]:
        """Charge la liste des classes"""
        classes_file = self.meta_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            # Si classes.txt n'existe pas, lister les dossiers
            classes = sorted([d.name for d in self.images_dir.iterdir() if d.is_dir()])

        return classes

    def _load_split(self) -> Tuple[List[Path], List[int]]:
        """Charge les chemins d'images et labels pour le split"""
        split_file = self.meta_dir / f"{self.split}.txt"

        image_paths = []
        labels = []

        with open(split_file, 'r') as f:
            for line in f:
                # Format: class_name/image_name
                rel_path = line.strip()
                class_name = rel_path.split('/')[0]

                # Chemin complet
                img_path = self.images_dir / f"{rel_path}.jpg"

                if img_path.exists():
                    image_paths.append(img_path)
                    labels.append(self.class_to_idx[class_name])

        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retourne (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Charger l'image
        image = Image.open(img_path).convert('RGB')

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Retourne le nom de la classe pour un index"""
        return self.idx_to_class[idx]

    def get_class_distribution(self) -> dict:
        """Retourne la distribution des classes"""
        from collections import Counter
        distribution = Counter(self.labels)
        return {self.idx_to_class[idx]: count for idx, count in distribution.items()}


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    pin_memory: bool = True
):
    """
    Crée les DataLoaders pour train et test

    Args:
        root_dir: Chemin vers food-101/
        batch_size: Taille des batchs
        num_workers: Nombre de workers pour le chargement
        train_transform: Transformations pour l'entraînement
        test_transform: Transformations pour le test
        pin_memory: Utiliser pin_memory pour GPU

    Returns:
        train_loader, test_loader, num_classes
    """
    from torch.utils.data import DataLoader

    # Créer les datasets
    train_dataset = Food101Dataset(root_dir, split='train', transform=train_transform)
    test_dataset = Food101Dataset(root_dir, split='test', transform=test_transform)

    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Pour stabilité du batch norm
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    num_classes = len(train_dataset.classes)

    print(f"✓ DataLoaders créés:")
    print(f"  - Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  - Test: {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")

    return train_loader, test_loader, num_classes


if __name__ == "__main__":
    # Test du dataset
    from transforms import get_transforms

    root_dir = "../../data/food-101"
    train_transform, test_transform = get_transforms()

    # Test train dataset
    train_dataset = Food101Dataset(root_dir, split='train', transform=train_transform)
    print(f"\nDataset d'entraînement: {len(train_dataset)} images")

    # Test d'un échantillon
    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label} ({train_dataset.get_class_name(label)})")

    # Distribution
    print("\nDistribution des classes (premières 5):")
    dist = train_dataset.get_class_distribution()
    for class_name, count in list(dist.items())[:5]:
        print(f"  {class_name}: {count}")
