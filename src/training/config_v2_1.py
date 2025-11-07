"""
Configuration OPTIMISÉE V2.1 - Correctifs Rapides
Objectif: 75-78% Top-1 Accuracy (vs 66% actuel)
Durée estimée: ~25h sur T4 GPU

CHANGEMENTS vs V2:
1. CUTMIX_ALPHA réduit: 1.0 → 0.3 (moins agressif)
2. MIXUP_PROB réduit: 0.5 → 0.3 (70% images normales)
3. PHASE2_LR ajusté: 1e-4 → 7.5e-5 (convergence plus fine)
4. PHASE2_EPOCHS augmenté: 80 → 100 (+20 epochs)
5. AUGMENTATION_LEVEL: heavy → medium (moins de déformations)

GAINS ATTENDUS:
- Top-1 Accuracy: 66% → 75-78% (+9 à +12 points)
- Top-5 Accuracy: 89% → 94-96% (+5 à +7 points)
"""

from pathlib import Path


class ConfigV2_1:
    """Configuration optimisée V2.1 pour amélioration rapide"""

    # ============ Chemins ============
    DATA_DIR = Path("data/food-101")
    CHECKPOINT_DIR = Path("checkpoints_v2_1")
    RESULTS_DIR = Path("results_v2_1")

    # Créer les dossiers si nécessaire
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ============ Modèle ============
    NUM_CLASSES = 101
    PRETRAINED = True
    DROPOUT = 0.2  # Maintenu (déjà optimal)

    # ============ Données ============
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Normalisation ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Data augmentation - RÉDUIT DE HEAVY → MEDIUM
    AUGMENTATION_LEVEL = 'medium'  # ✅ OPTIMISÉ: Moins agressif pour faciliter l'apprentissage

    # ============ Entraînement Phase 1 ============
    PHASE1_EPOCHS = 5
    PHASE1_LR = 1e-3
    PHASE1_OPTIMIZER = 'adam'
    PHASE1_WEIGHT_DECAY = 1e-4

    # ============ Entraînement Phase 2 ============
    PHASE2_EPOCHS = 100  # ✅ OPTIMISÉ: Augmenté de 80 → 100 pour convergence complète
    PHASE2_LR = 7.5e-5  # ✅ OPTIMISÉ: Réduit de 1e-4 → 7.5e-5 pour convergence plus fine
    PHASE2_OPTIMIZER = 'sgd'
    PHASE2_MOMENTUM = 0.9
    PHASE2_WEIGHT_DECAY = 1e-4

    # ============ Scheduler ============
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'
    STEP_SIZE = 3
    GAMMA = 0.1
    T_MAX = 100  # ✅ OPTIMISÉ: Aligné avec PHASE2_EPOCHS
    WARMUP_EPOCHS = 5

    # ============ Training ============
    USE_AMP = True
    GRADIENT_CLIP = 1.0

    # Advanced augmentation - RÉDUIT POUR MOINS D'AGRESSIVITÉ
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2  # Maintenu (déjà bon)
    USE_CUTMIX = True
    CUTMIX_ALPHA = 0.3  # ✅ OPTIMISÉ: Réduit de 1.0 → 0.3 (standard pour Food-101)
    MIXUP_PROB = 0.3  # ✅ OPTIMISÉ: Réduit de 0.5 → 0.3 (70% images normales)

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15  # ✅ OPTIMISÉ: Augmenté de 12 → 15 (plus de patience)
    EARLY_STOPPING_DELTA = 0.001

    # Label Smoothing
    LABEL_SMOOTHING = 0.1  # Maintenu (bon)

    # Sauvegarde
    SAVE_BEST_ONLY = True
    SAVE_EVERY_N_EPOCHS = 5

    # ============ Évaluation ============
    EVAL_EVERY_N_EPOCHS = 1
    TOPK = (1, 5)

    # ============ Device ============
    DEVICE = 'cuda'

    # ============ Logging ============
    PRINT_FREQ = 50
    USE_WANDB = False
    WANDB_PROJECT = "food101-classifier-v2.1"
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
        print("CONFIGURATION V2.1 - CORRECTIFS RAPIDES")
        print("="*80)

        print("\n📁 CHEMINS:")
        print(f"  - Data dir: {cls.DATA_DIR}")
        print(f"  - Checkpoint dir: {cls.CHECKPOINT_DIR}")
        print(f"  - Results dir: {cls.RESULTS_DIR}")

        print("\n🧠 MODÈLE:")
        print(f"  - Architecture: ResNet-50")
        print(f"  - Classes: {cls.NUM_CLASSES}")
        print(f"  - Pretrained: {cls.PRETRAINED}")
        print(f"  - Dropout: {cls.DROPOUT}")

        print("\n📊 DONNÉES:")
        print(f"  - Image size: {cls.IMG_SIZE}")
        print(f"  - Batch size: {cls.BATCH_SIZE}")
        print(f"  - Num workers: {cls.NUM_WORKERS}")
        print(f"  - Augmentation: {cls.AUGMENTATION_LEVEL.upper()} ⬇️ (RÉDUIT: heavy→medium)")

        print("\n🏋️ ENTRAÎNEMENT:")
        print(f"  - Phase 1: {cls.PHASE1_EPOCHS} epochs @ LR={cls.PHASE1_LR}")
        print(f"  - Phase 2: {cls.PHASE2_EPOCHS} epochs @ LR={cls.PHASE2_LR} ⬆️ (OPTIMISÉ: 80→100 epochs, LR 1e-4→7.5e-5)")
        print(f"  - Total epochs: {cls.get_total_epochs()} (85→105)")
        print(f"  - Optimizer Phase 1: {cls.PHASE1_OPTIMIZER.upper()}")
        print(f"  - Optimizer Phase 2: {cls.PHASE2_OPTIMIZER.upper()}")

        print("\n⚡ OPTIMISATIONS V2.1:")
        print(f"  - Mixed Precision (AMP): {cls.USE_AMP}")
        print(f"  - Gradient Clipping: {cls.GRADIENT_CLIP}")
        print(f"  - Scheduler: {cls.SCHEDULER_TYPE.upper()} (T_max={cls.T_MAX})")
        print(f"  - Label Smoothing: {cls.LABEL_SMOOTHING}")
        print(f"  - MixUp: {cls.USE_MIXUP} (alpha={cls.MIXUP_ALPHA})")
        print(f"  - CutMix: {cls.USE_CUTMIX} (alpha={cls.CUTMIX_ALPHA}) ⬇️ (RÉDUIT: 1.0→0.3)")
        print(f"  - MixUp/CutMix prob: {cls.MIXUP_PROB*100}% ⬇️ (RÉDUIT: 50%→30%)")

        print("\n💾 SAUVEGARDE:")
        print(f"  - Save best only: {cls.SAVE_BEST_ONLY}")
        print(f"  - Early stopping patience: {cls.EARLY_STOPPING_PATIENCE} epochs")
        print(f"  - Save every N epochs: {cls.SAVE_EVERY_N_EPOCHS}")

        print("\n🖥️ DEVICE:")
        print(f"  - Device: {cls.DEVICE}")

        print("\n" + "="*80)
        print("⏱️ DURÉE ESTIMÉE: 24-28 heures sur T4 GPU")
        print("🎯 OBJECTIF: Top-1 Accuracy = 75-78%")
        print("📈 AMÉLIORATION vs V2: +9 à +12% (66% → 75-78%)")
        print("="*80)

    @classmethod
    def get_changes_summary(cls):
        """Résumé des changements vs V2"""
        changes = [
            ("AUGMENTATION_LEVEL", "heavy", "medium", "Moins de déformations"),
            ("CUTMIX_ALPHA", "1.0", "0.3", "Mélange moins agressif"),
            ("MIXUP_PROB", "0.5 (50%)", "0.3 (30%)", "Plus d'images normales"),
            ("PHASE2_LR", "1e-4", "7.5e-5", "Convergence plus fine"),
            ("PHASE2_EPOCHS", "80", "100", "+20 epochs"),
            ("T_MAX", "80", "100", "Scheduler aligné"),
            ("EARLY_STOPPING_PATIENCE", "12", "15", "Plus de patience"),
        ]

        print("\n" + "="*80)
        print("📊 RÉSUMÉ DES CHANGEMENTS V2 → V2.1")
        print("="*80)
        print(f"\n{'Paramètre':<30} {'V2':<15} {'V2.1':<15} {'Justification':<30}")
        print("-"*80)
        for param, v2, v2_1, justif in changes:
            print(f"{param:<30} {v2:<15} {v2_1:<15} {justif:<30}")
        print("="*80)


# Configuration pour debug
class DebugConfigV2_1(ConfigV2_1):
    """Configuration pour tests rapides"""
    PHASE1_EPOCHS = 1
    PHASE2_EPOCHS = 2
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    PRINT_FREQ = 10


if __name__ == "__main__":
    # Afficher la configuration
    ConfigV2_1.print_config()
    print("\n")
    ConfigV2_1.get_changes_summary()
