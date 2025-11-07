"""
EfficientNet-B4 Classifier pour Food-101
Architecture SOTA optimisée pour la classification d'images
"""

import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    """
    Classificateur basé sur EfficientNet-B4

    EfficientNet utilise compound scaling pour optimiser:
    - Depth (profondeur du réseau)
    - Width (largeur des couches)
    - Resolution (taille des images)

    EfficientNet-B4:
    - Input: 380x380x3
    - Paramètres: ~19M
    - Top-1 ImageNet: 83%
    - SOTA Food-101: ~92%
    """

    def __init__(self, num_classes=101, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes (int): Nombre de classes de sortie
            pretrained (bool): Utiliser les poids ImageNet
            dropout (float): Taux de dropout avant la classification
        """
        super(EfficientNetClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Charger EfficientNet-B4 pré-entraîné via timm
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Pas de tête de classification
            global_pool=''  # On gère le pooling nous-mêmes
        )

        # Nombre de features en sortie du backbone
        self.num_features = self.backbone.num_features  # 1792 pour EfficientNet-B4

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes)
        )

        # Initialiser la tête de classification
        self._init_classifier()

    def _init_classifier(self):
        """Initialise les poids de la tête de classification"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input images [B, 3, 380, 380]

        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        # Extraction de features
        features = self.backbone(x)  # [B, 1792, H, W]

        # Global pooling
        features = self.global_pool(features)  # [B, 1792, 1, 1]
        features = features.flatten(1)  # [B, 1792]

        # Classification
        logits = self.classifier(features)  # [B, num_classes]

        return logits

    def freeze_backbone(self):
        """Gèle le backbone (pour Phase 1)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone EfficientNet-B4 gelé (Phase 1)")

    def unfreeze_backbone(self):
        """Dégèle le backbone (pour Phase 2)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone EfficientNet-B4 dégelé (Phase 2)")

    def get_num_params(self):
        """Retourne le nombre de paramètres du modèle"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def print_model_info(self):
        """Affiche les informations du modèle"""
        total, trainable = self.get_num_params()
        print("\n" + "="*60)
        print("MODÈLE: EfficientNet-B4 Classifier")
        print("="*60)
        print(f"Architecture: EfficientNet-B4")
        print(f"Input size: 380×380×3")
        print(f"Nombre de classes: {self.num_classes}")
        print(f"Dropout: {self.dropout_rate}")
        print(f"\nParamètres:")
        print(f"  - Total: {total:,} ({total/1e6:.1f}M)")
        print(f"  - Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
        print(f"  - Features: {self.num_features}")
        print("="*60 + "\n")


def create_efficientnet_model(num_classes=101, pretrained=True, dropout=0.3, device='cuda'):
    """
    Fonction helper pour créer un modèle EfficientNet-B4

    Args:
        num_classes (int): Nombre de classes
        pretrained (bool): Utiliser poids ImageNet
        dropout (float): Taux de dropout
        device (str): Device ('cuda' ou 'cpu')

    Returns:
        EfficientNetClassifier: Modèle initialisé
    """
    model = EfficientNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    # Déplacer vers le device
    model = model.to(device)

    # Afficher les infos
    model.print_model_info()

    return model


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Charge un checkpoint

    Args:
        model (EfficientNetClassifier): Modèle
        checkpoint_path (str): Chemin du checkpoint
        device (str): Device

    Returns:
        dict: Informations du checkpoint (epoch, accuracy, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Checkpoint chargé: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"   Best Accuracy: {checkpoint['best_acc']:.2f}%")

    return checkpoint


if __name__ == "__main__":
    # Test du modèle
    print("Test EfficientNet-B4 Classifier\n")

    # Créer le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model = create_efficientnet_model(
        num_classes=101,
        pretrained=True,
        dropout=0.3,
        device=device
    )

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 380, 380).to(device)

    print(f"Test forward pass:")
    print(f"  Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test freeze/unfreeze
    print(f"\nTest freeze/unfreeze:")
    model.freeze_backbone()
    total, trainable = model.get_num_params()
    print(f"  Trainable params (Phase 1): {trainable:,} ({trainable/1e6:.1f}M)")

    model.unfreeze_backbone()
    total, trainable = model.get_num_params()
    print(f"  Trainable params (Phase 2): {trainable:,} ({trainable/1e6:.1f}M)")

    print("\n✅ Test réussi!")
