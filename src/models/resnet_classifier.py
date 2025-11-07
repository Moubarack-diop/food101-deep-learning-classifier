"""
Modèle ResNet-50 pour classification Food-101
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 avec transfer learning pour Food-101

    Architecture:
        - Backbone: ResNet-50 pré-entraîné sur ImageNet
        - Head: Fully Connected avec Dropout
        - Output: 101 classes

    Args:
        num_classes: Nombre de classes (101 pour Food-101)
        pretrained: Utiliser les poids ImageNet
        dropout: Probabilité de dropout avant la couche finale
        freeze_backbone: Geler le backbone pour Phase 1
    """

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(ResNet50Classifier, self).__init__()

        # Charger ResNet-50 pré-entraîné
        self.backbone = models.resnet50(pretrained=pretrained)

        # Nombre de features avant la couche FC
        num_features = self.backbone.fc.in_features

        # Remplacer la tête de classification
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )

        # Optionnel: Geler le backbone (Phase 1)
        if freeze_backbone:
            self.freeze_backbone()

        # Sauvegarder les paramètres
        self.num_classes = num_classes
        self.num_features = num_features

        print(f"✓ ResNet50Classifier créé:")
        print(f"  - Pré-entraîné: {pretrained}")
        print(f"  - Nombre de classes: {num_classes}")
        print(f"  - Features: {num_features}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Backbone gelé: {freeze_backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Tensor (B, 3, 224, 224)

        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """Gèle tous les paramètres sauf la tête de classification (Phase 1)"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Ne pas geler la couche FC
                param.requires_grad = False

        print("✓ Backbone gelé (seule la tête sera entraînée)")

    def unfreeze_backbone(self):
        """Dégèle tout le réseau pour fine-tuning (Phase 2)"""
        for param in self.backbone.parameters():
            param.requires_grad = True

        print("✓ Backbone dégelé (fine-tuning complet)")

    def get_trainable_params(self):
        """Retourne le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Retourne le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())

    def print_param_info(self):
        """Affiche les informations sur les paramètres"""
        trainable = self.get_trainable_params()
        total = self.get_total_params()

        print(f"\n Paramètres du modèle:")
        print(f"  - Total: {total:,}")
        print(f"  - Entraînables: {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"  - Gelés: {total - trainable:,} ({100 * (total - trainable) / total:.1f}%)")


def create_model(
    num_classes: int = 101,
    pretrained: bool = True,
    dropout: float = 0.5,
    phase: int = 1
) -> ResNet50Classifier:
    """
    Fonction helper pour créer le modèle

    Args:
        num_classes: Nombre de classes
        pretrained: Utiliser les poids ImageNet
        dropout: Taux de dropout
        phase: Phase d'entraînement (1: backbone gelé, 2: fine-tuning)

    Returns:
        Modèle ResNet50Classifier
    """
    freeze = (phase == 1)
    model = ResNet50Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze
    )

    model.print_param_info()
    return model


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 101,
    device: str = 'cuda'
) -> ResNet50Classifier:
    """
    Charge un modèle depuis un checkpoint

    Args:
        checkpoint_path: Chemin vers le fichier .pth
        num_classes: Nombre de classes
        device: Device ('cuda' ou 'cpu')

    Returns:
        Modèle chargé
    """
    # Créer le modèle
    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)

    # Charger les poids (weights_only=False pour compatibilité PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"✓ Modèle chargé depuis {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Test du modèle
    print("=== Test ResNet50Classifier ===\n")

    # Phase 1: Backbone gelé
    print("Phase 1: Entraînement de la tête uniquement")
    model_phase1 = create_model(phase=1)

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model_phase1(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Phase 2: Fine-tuning complet
    print("\n" + "="*50)
    print("\nPhase 2: Fine-tuning complet")
    model_phase2 = create_model(phase=2)

    # Vérification GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_phase2 = model_phase2.to(device)
        dummy_input = dummy_input.to(device)
        output = model_phase2(dummy_input)
        print(f"\n✓ Modèle testé sur GPU")
        print(f"  Device: {output.device}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("\n GPU non disponible, test sur CPU uniquement")

    print("\n Tests terminés avec succès!")
