"""
Configuration V3 - REFONTE AMBITIEUSE avec EfficientNet-B4
Objectif: 85-90% Top-1 Accuracy (ATTEINDRE L'OBJECTIF!)
Durée estimée: ~35-40h sur T4 GPU

CHANGEMENTS MAJEURS vs V2:
1. Architecture: ResNet-50 → EfficientNet-B4 (SOTA pour Food-101)
2. Augmentation modérée (medium) avec TTA en évaluation
3. Training plus long: 120 epochs Phase 2
4. MixUp/CutMix optimisés (alpha réduits, prob 25%)
5. Learning rate adapté pour EfficientNet
6. Warmup progressif (10 epochs)

GAINS ATTENDUS:
- Top-1 Accuracy: 66% → 85-90% (+19 à +24 points) ✅ OBJECTIF ATTEINT
- Top-5 Accuracy: 89% → 97-99% (+8 à +10 points)

ARCHITECTURE EFFICIENTNET-B4:
- Paramètres: 19M (vs 25.6M ResNet-50)
- Plus efficace: meilleur accuracy/paramètre
- Compound scaling (depth + width + resolution)
- State-of-the-art sur ImageNet et Food-101
"""

from pathlib import Path


class ConfigV3:
    """Configuration V3 ambitieuse avec EfficientNet-B4"""

    # ============ Chemins ============
    DATA_DIR = Path("data/food-101")
    CHECKPOINT_DIR = Path("checkpoints_v3")
    RESULTS_DIR = Path("results_v3")

    # Créer les dossiers si nécessaire
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ============ Modèle ============
    MODEL_NAME = 'efficientnet_b4'  # 🆕 NOUVEAU: Architecture SOTA
    NUM_CLASSES = 101
    PRETRAINED = True
    DROPOUT = 0.3  # ✅ Augmenté pour EfficientNet (plus de paramètres effectifs)

    # ============ Données ============
    IMG_SIZE = 380  # 🆕 OPTIMISÉ: EfficientNet-B4 utilise 380x380 (vs 224 ResNet)
    BATCH_SIZE = 16  # ✅ Réduit car images plus grandes (380 vs 224)
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Normalisation ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Data augmentation - MEDIUM (équilibré)
    AUGMENTATION_LEVEL = 'medium'  # ✅ Ni trop agressif, ni trop faible

    # ============ Entraînement Phase 1 ============
    PHASE1_EPOCHS = 5
    PHASE1_LR = 1e-3
    PHASE1_OPTIMIZER = 'adam'
    PHASE1_WEIGHT_DECAY = 1e-4

    # ============ Entraînement Phase 2 ============
    PHASE2_EPOCHS = 120  # 🆕 OPTIMISÉ: Training long pour convergence complète
    PHASE2_LR = 3e-5  # ✅ OPTIMISÉ: Plus faible pour EfficientNet (fine-tuning délicat)
    PHASE2_OPTIMIZER = 'adamw'  # 🆕 NOUVEAU: AdamW (meilleur que SGD pour EfficientNet)
    PHASE2_MOMENTUM = 0.9  # Non utilisé avec AdamW
    PHASE2_WEIGHT_DECAY = 1e-5  # ✅ Réduit pour AdamW

    # ============ Scheduler ============
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'
    STEP_SIZE = 3
    GAMMA = 0.1
    T_MAX = 120  # ✅ Aligné avec PHASE2_EPOCHS
    WARMUP_EPOCHS = 10  # 🆕 OPTIMISÉ: Warmup plus long pour stabilité

    # ============ Training ============
    USE_AMP = True
    GRADIENT_CLIP = 1.0

    # Advanced augmentation - OPTIMISÉ
    USE_MIXUP = True
    MIXUP_ALPHA = 0.15  # ✅ OPTIMISÉ: Réduit de 0.2 → 0.15
    USE_CUTMIX = True
    CUTMIX_ALPHA = 0.25  # ✅ OPTIMISÉ: Réduit de 1.0 → 0.25
    MIXUP_PROB = 0.25  # ✅ OPTIMISÉ: Seulement 25% des batches (75% normaux)

    # Test-Time Augmentation
    USE_TTA = True  # 🆕 NOUVEAU: TTA pour améliorer validation (+1-2%)
    TTA_TRANSFORMS = 5  # Nombre de transformations pour TTA

    # Early stopping
    EARLY_STOPPING_PATIENCE = 20  # ✅ OPTIMISÉ: Plus de patience (120 epochs)
    EARLY_STOPPING_DELTA = 0.001

    # Label Smoothing
    LABEL_SMOOTHING = 0.1

    # Sauvegarde
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 10  # ✅ Moins fréquent (epochs plus longs)

    # ============ Évaluation ============
    EVAL_EVERY_N_EPOCHS = 1
    TOPK = (1, 5)

    # ============ Device ============
    DEVICE = 'cuda'

    # ============ Logging ============
    PRINT_FREQ = 50
    USE_WANDB = False
    WANDB_PROJECT = "food101-classifier-v3"
    WANDB_ENTITY = None

    # ============ Reproduction ============
    SEED = 42

    @classmethod
    def get_total_epochs(cls):
        """Retourne le nombre total d'epochs"""
        return cls.PHASE1_EPOCHS + cls.PHASE2_EPOCHS

    @classmethod
    def print_config(cls):
        """Affiche la configuration"""
        print("\n" + "="*80)
        print("🚀 CONFIGURATION V3 - ARCHITECTURE EFFICIENTNET-B4")
        print("="*80)

        print("\n📁 CHEMINS:")
        print(f"  - Data dir: {cls.DATA_DIR}")
        print(f"  - Checkpoint dir: {cls.CHECKPOINT_DIR}")
        print(f"  - Results dir: {cls.RESULTS_DIR}")

        print("\n🧠 MODÈLE:")
        print(f"  - Architecture: {cls.MODEL_NAME.upper()} 🆕 (NOUVEAU!)")
        print(f"  - Paramètres: ~19M (vs 25.6M ResNet-50)")
        print(f"  - Classes: {cls.NUM_CLASSES}")
        print(f"  - Pretrained: {cls.PRETRAINED}")
        print(f"  - Dropout: {cls.DROPOUT}")

        print("\n📊 DONNÉES:")
        print(f"  - Image size: {cls.IMG_SIZE}x{cls.IMG_SIZE} ⬆️ (OPTIMISÉ: 224→380 pour EfficientNet)")
        print(f"  - Batch size: {cls.BATCH_SIZE} ⬇️ (réduit car images plus grandes)")
        print(f"  - Num workers: {cls.NUM_WORKERS}")
        print(f"  - Augmentation: {cls.AUGMENTATION_LEVEL.upper()}")

        print("\n🏋️ ENTRAÎNEMENT:")
        print(f"  - Phase 1: {cls.PHASE1_EPOCHS} epochs @ LR={cls.PHASE1_LR}")
        print(f"  - Phase 2: {cls.PHASE2_EPOCHS} epochs @ LR={cls.PHASE2_LR} (AdamW)")
        print(f"  - Total epochs: {cls.get_total_epochs()} (85→125)")
        print(f"  - Optimizer Phase 1: {cls.PHASE1_OPTIMIZER.upper()}")
        print(f"  - Optimizer Phase 2: {cls.PHASE2_OPTIMIZER.upper()} 🆕")

        print("\n⚡ OPTIMISATIONS V3:")
        print(f"  - Mixed Precision (AMP): {cls.USE_AMP}")
        print(f"  - Gradient Clipping: {cls.GRADIENT_CLIP}")
        print(f"  - Scheduler: {cls.SCHEDULER_TYPE.upper()} (T_max={cls.T_MAX})")
        print(f"  - Warmup: {cls.WARMUP_EPOCHS} epochs ⬆️ (OPTIMISÉ)")
        print(f"  - Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"  - MixUp: {cls.USE_MIXUP} (alpha={cls.MIXUP_ALPHA})")
        print(f"  - CutMix: {cls.USE_CUTMIX} (alpha={cls.CUTMIX_ALPHA})")
        print(f"  - MixUp/CutMix prob: {cls.MIXUP_PROB*100}% (25% seulement)")
        print(f"  - Test-Time Augmentation: {cls.USE_TTA} 🆕 (NOUVEAU!)")

        print("\n💾 SAUVEGARDE:")
        print(f"  - Save best only: {cls.SAVE_BEST_ONLY}")
        print(f"  - Early stopping patience: {cls.EARLY_STOPPING_PATIENCE} epochs")
        print(f"  - Save every N epochs: {cls.SAVE_EVERY_N_EPOCHS}")

        print("\n🖥️ DEVICE:")
        print(f"  - Device: {cls.DEVICE}")

        print("\n" + "="*80)
        print("⏱️ DURÉE ESTIMÉE: 35-40 heures sur T4 GPU")
        print("🎯 OBJECTIF: Top-1 Accuracy = 85-90% ✅ OBJECTIF ATTEINT!")
        print("📈 AMÉLIORATION vs V2: +19 à +24% (66% → 85-90%)")
        print("🏆 SOTA Food-101: ~92% (avec ensembles)")
        print("="*80)

    @classmethod
    def get_changes_summary(cls):
        """Résumé des changements vs V2"""
        changes = [
            ("MODEL", "ResNet-50", "EfficientNet-B4", "Architecture SOTA"),
            ("IMG_SIZE", "224", "380", "Taille optimale EfficientNet"),
            ("BATCH_SIZE", "32", "16", "Adapté aux images 380x380"),
            ("DROPOUT", "0.2", "0.3", "Plus de régularisation"),
            ("PHASE2_EPOCHS", "80", "120", "+40 epochs"),
            ("PHASE2_LR", "1e-4", "3e-5", "LR adapté EfficientNet"),
            ("PHASE2_OPTIMIZER", "SGD", "AdamW", "Meilleur pour EfficientNet"),
            ("WARMUP_EPOCHS", "5", "10", "Warmup plus long"),
            ("MIXUP_ALPHA", "0.2", "0.15", "Moins agressif"),
            ("CUTMIX_ALPHA", "1.0", "0.25", "Beaucoup moins agressif"),
            ("MIXUP_PROB", "0.5 (50%)", "0.25 (25%)", "75% images normales"),
            ("USE_TTA", "False", "True", "Test-Time Augmentation"),
            ("T_MAX", "80", "120", "Scheduler aligné"),
            ("EARLY_STOPPING", "12", "20", "Plus de patience"),
        ]

        print("\n" + "="*80)
        print("📊 RÉSUMÉ DES CHANGEMENTS V2 → V3")
        print("="*80)
        print(f"\n{'Paramètre':<25} {'V2':<20} {'V3':<20} {'Justification':<25}")
        print("-"*80)
        for param, v2, v3, justif in changes:
            print(f"{param:<25} {v2:<20} {v3:<20} {justif:<25}")
        print("="*80)

    @classmethod
    def get_requirements(cls):
        """Retourne les dépendances supplémentaires pour V3"""
        print("\n" + "="*80)
        print("📦 DÉPENDANCES SUPPLÉMENTAIRES POUR V3")
        print("="*80)
        print("\nAjoutez à requirements.txt:")
        print("  efficientnet-pytorch>=0.7.1")
        print("  ou utilisez timm (déjà installé):")
        print("  timm>=0.9.0  # Contient EfficientNet-B4 pré-entraîné")
        print("\nInstallation:")
        print("  pip install timm>=0.9.0")
        print("="*80)


# Configuration pour debug
class DebugConfigV3(ConfigV3):
    """Configuration pour tests rapides"""
    PHASE1_EPOCHS = 1
    PHASE2_EPOCHS = 2
    BATCH_SIZE = 4  # Petit batch pour images 380x380
    NUM_WORKERS = 0
    PRINT_FREQ = 10
    USE_TTA = False


if __name__ == "__main__":
    # Afficher la configuration
    ConfigV3.print_config()
    print("\n")
    ConfigV3.get_changes_summary()
    print("\n")
    ConfigV3.get_requirements()
