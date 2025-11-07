"""
Configuration pour l'entraînement du modèle Food-101
"""

from pathlib import Path


class Config:
    """Configuration centralisée pour l'entraînement"""

    # ============ Chemins ============
    DATA_DIR = Path("data/food-101")
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")

    # Créer les dossiers si nécessaire
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ============ Modèle ============
    NUM_CLASSES = 101
    PRETRAINED = True
    DROPOUT = 0.2  # OPTIMISÉ: Réduit de 0.3 à 0.2 pour réduire la régularisation excessive

    # ============ Données ============
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Normalisation ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Data augmentation
    AUGMENTATION_LEVEL = 'heavy'  # 'light', 'medium', 'heavy' - Augmentation plus agressive pour dataset "in the wild"

    # ============ Entraînement Phase 1 ============
    # Phase 1: Entraînement de la tête uniquement (backbone gelé)
    PHASE1_EPOCHS = 5  # Augmenté de 3 à 5 pour meilleure adaptation de la tête
    PHASE1_LR = 1e-3
    PHASE1_OPTIMIZER = 'adam'  # 'adam' ou 'sgd'
    PHASE1_WEIGHT_DECAY = 1e-4

    # ============ Entraînement Phase 2 ============
    # Phase 2: Fine-tuning complet du réseau
    PHASE2_EPOCHS = 80  # OPTIMISÉ: Augmenté de 30 à 80 pour convergence complète
    PHASE2_LR = 1e-4  # OPTIMISÉ: Augmenté de 5e-5 à 1e-4 pour convergence plus rapide
    PHASE2_OPTIMIZER = 'sgd'  # 'adam' ou 'sgd'
    PHASE2_MOMENTUM = 0.9
    PHASE2_WEIGHT_DECAY = 1e-4

    # ============ Scheduler ============
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # 'step', 'cosine', 'plateau' - Cosine pour décroissance smooth du LR
    STEP_SIZE = 3  # Pour StepLR
    GAMMA = 0.1  # Pour StepLR
    T_MAX = 80  # OPTIMISÉ: Pour CosineAnnealingLR - doit correspondre à PHASE2_EPOCHS
    WARMUP_EPOCHS = 5  # OPTIMISÉ: Nombre d'epochs de warmup pour le learning rate

    # ============ Training ============
    USE_AMP = True  # Mixed Precision Training
    GRADIENT_CLIP = 1.0  # Gradient clipping (None pour désactiver)

    # Advanced augmentation techniques
    USE_MIXUP = True  # Activer MixUp augmentation
    MIXUP_ALPHA = 0.2  # Paramètre alpha pour MixUp
    USE_CUTMIX = True  # Activer CutMix augmentation
    CUTMIX_ALPHA = 1.0  # Paramètre alpha pour CutMix
    MIXUP_PROB = 0.5  # Probabilité d'appliquer MixUp/CutMix (50% MixUp, 50% CutMix)

    # Early stopping
    EARLY_STOPPING_PATIENCE = 12  # OPTIMISÉ: Augmenté de 7 à 12 pour plus de patience
    EARLY_STOPPING_DELTA = 0.001

    # Label Smoothing
    LABEL_SMOOTHING = 0.1  # NOUVEAU: Label smoothing pour régularisation

    # Sauvegarde
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 5

    # ============ Évaluation ============
    EVAL_EVERY_N_EPOCHS = 1
    TOPK = (1, 5)  # Top-k accuracies à calculer

    # ============ Device ============
    DEVICE = 'cuda'  # 'cuda' ou 'cpu'

    # ============ Logging ============
    PRINT_FREQ = 50  # Afficher les logs tous les N batches
    USE_WANDB = False  # Utiliser Weights & Biases
    WANDB_PROJECT = "food101-classifier"
    WANDB_ENTITY = None  # Votre username W&B

    # ============ Reproduction ============
    SEED = 42

    @classmethod
    def get_total_epochs(cls):
        """Retourne le nombre total d'epochs"""
        return cls.PHASE1_EPOCHS + cls.PHASE2_EPOCHS

    @classmethod
    def print_config(cls):
        """Affiche la configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION D'ENTRAÎNEMENT")
        print("="*60)

        print("\n CHEMINS:")
        print(f"  Data dir: {cls.DATA_DIR}")
        print(f"  Checkpoint dir: {cls.CHECKPOINT_DIR}")
        print(f"  Results dir: {cls.RESULTS_DIR}")

        print("\n  MODÈLE:")
        print(f"  Classes: {cls.NUM_CLASSES}")
        print(f"  Pretrained: {cls.PRETRAINED}")
        print(f"  Dropout: {cls.DROPOUT}")

        print("\n DONNÉES:")
        print(f"  Image size: {cls.IMG_SIZE}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Num workers: {cls.NUM_WORKERS}")
        print(f"  Augmentation: {cls.AUGMENTATION_LEVEL}")

        print("\n ENTRAÎNEMENT:")
        print(f"  Phase 1 (tête): {cls.PHASE1_EPOCHS} epochs @ LR={cls.PHASE1_LR}")
        print(f"  Phase 2 (fine-tune): {cls.PHASE2_EPOCHS} epochs @ LR={cls.PHASE2_LR}")
        print(f"  Total epochs: {cls.get_total_epochs()}")
        print(f"  Mixed Precision: {cls.USE_AMP}")
        print(f"  Scheduler: {cls.SCHEDULER_TYPE if cls.USE_SCHEDULER else 'None'}")
        print(f"  MixUp: {cls.USE_MIXUP} (alpha={cls.MIXUP_ALPHA if cls.USE_MIXUP else 'N/A'})")
        print(f"  CutMix: {cls.USE_CUTMIX} (alpha={cls.CUTMIX_ALPHA if cls.USE_CUTMIX else 'N/A'})")

        print("\n SAUVEGARDE:")
        print(f"  Save best only: {cls.SAVE_BEST_ONLY}")
        print(f"  Early stopping: {cls.EARLY_STOPPING_PATIENCE} epochs")

        print("\n  DEVICE:")
        print(f"  Device: {cls.DEVICE}")

        print("="*60 + "\n")


# Configuration pour tests rapides (développement)
class DebugConfig(Config):
    """Configuration pour debug/test rapide"""
    PHASE1_EPOCHS = 1
    PHASE2_EPOCHS = 2
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    PRINT_FREQ = 10


if __name__ == "__main__":
    # Afficher la configuration
    Config.print_config()

    print("\n Configuration de debug:")
    DebugConfig.print_config()
