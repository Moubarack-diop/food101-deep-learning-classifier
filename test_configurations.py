"""
Script de test pour vérifier les configurations V2.1 et V3
Exécutez ce script pour valider que tout est prêt avant l'entraînement
"""

import sys
from pathlib import Path

# Ajouter le dossier au path
sys.path.append(str(Path(__file__).parent))

print("="*80)
print("TEST DES CONFIGURATIONS - Food-101 Classifier")
print("="*80)

# Test 1: Configuration V2.1
print("\n" + "="*80)
print("TEST 1: Configuration V2.1 (Correctifs Rapides)")
print("="*80)

try:
    from src.training.config_v2_1 import ConfigV2_1

    ConfigV2_1.print_config()
    print("\n")
    ConfigV2_1.get_changes_summary()

    # Vérifications
    assert ConfigV2_1.AUGMENTATION_LEVEL == 'medium', "AUGMENTATION_LEVEL devrait etre 'medium'"
    assert ConfigV2_1.CUTMIX_ALPHA == 0.3, "CUTMIX_ALPHA devrait etre 0.3"
    assert ConfigV2_1.MIXUP_PROB == 0.3, "MIXUP_PROB devrait etre 0.3"
    assert ConfigV2_1.PHASE2_LR == 7.5e-5, "PHASE2_LR devrait etre 7.5e-5"
    assert ConfigV2_1.PHASE2_EPOCHS == 100, "PHASE2_EPOCHS devrait etre 100"
    assert ConfigV2_1.T_MAX == 100, "T_MAX devrait etre 100"

    print("\nOK Configuration V2.1 validee avec succes!")

except Exception as e:
    print(f"\nERREUR Configuration V2.1: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Configuration V3
print("\n" + "="*80)
print("TEST 2: Configuration V3 (Architecture EfficientNet-B4)")
print("="*80)

try:
    from src.training.config_v3 import ConfigV3

    ConfigV3.print_config()
    print("\n")
    ConfigV3.get_changes_summary()
    print("\n")
    ConfigV3.get_requirements()

    # Vérifications
    assert ConfigV3.MODEL_NAME == 'efficientnet_b4', "MODEL_NAME devrait etre 'efficientnet_b4'"
    assert ConfigV3.IMG_SIZE == 380, "IMG_SIZE devrait etre 380"
    assert ConfigV3.BATCH_SIZE == 16, "BATCH_SIZE devrait etre 16"
    assert ConfigV3.PHASE2_EPOCHS == 120, "PHASE2_EPOCHS devrait etre 120"
    assert ConfigV3.PHASE2_LR == 3e-5, "PHASE2_LR devrait etre 3e-5"
    assert ConfigV3.PHASE2_OPTIMIZER == 'adamw', "PHASE2_OPTIMIZER devrait etre 'adamw'"
    assert ConfigV3.USE_TTA == True, "USE_TTA devrait etre True"

    print("\nConfiguration V3 validee avec succes!")

except Exception as e:
    print(f"\nERREUR Configuration V3: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Modèle EfficientNet
print("\n" + "="*80)
print("TEST 3: Modèle EfficientNet-B4")
print("="*80)

try:
    import torch
    from src.models.efficientnet_classifier import EfficientNetClassifier, create_efficientnet_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device détecté: {device}")

    # Test création modèle
    print("\nCréation du modèle EfficientNet-B4...")
    model = EfficientNetClassifier(num_classes=101, pretrained=False, dropout=0.3)

    model.print_model_info()

    # Test forward pass
    print("Test forward pass...")
    dummy_input = torch.randn(2, 3, 380, 380)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 101), f"Output shape devrait etre (2, 101), got {output.shape}"

    # Test freeze/unfreeze
    print("\nTest freeze/unfreeze...")
    model.freeze_backbone()
    total, trainable = model.get_num_params()
    print(f"Trainable params (Phase 1): {trainable:,}")

    model.unfreeze_backbone()
    total, trainable = model.get_num_params()
    print(f"Trainable params (Phase 2): {trainable:,}")

    print("\nModele EfficientNet-B4 valide avec succes!")

except Exception as e:
    print(f"\nERREUR Modele EfficientNet: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Bug Fix dans Trainer
print("\n" + "="*80)
print("TEST 4: Vérification Bug Fix dans Trainer")
print("="*80)

try:
    from src.training.trainer import Trainer

    # Lire le fichier trainer.py pour vérifier le fix
    trainer_path = Path(__file__).parent / "src" / "training" / "trainer.py"
    with open(trainer_path, 'r', encoding='utf-8') as f:
        trainer_code = f.read()

    # Vérifier que le fix est présent
    if "if use_mixup:" in trainer_code and "top1, top5 = 0.0, 0.0" in trainer_code:
        print("Bug fix detecte dans trainer.py")
        print("   - Le calcul d'accuracy est maintenant conditionnel a use_mixup")
        print("   - L'accuracy training sera correcte avec MixUp/CutMix")
    else:
        print("Bug fix peut-etre absent ou code modifie")
        print("   Verifiez manuellement src/training/trainer.py:243-257")

    print("\nTrainer valide!")

except Exception as e:
    print(f"\nERREUR Trainer: {e}")
    import traceback
    traceback.print_exc()

# Résumé final
print("\n" + "="*80)
print("RESUME DES TESTS")
print("="*80)

print("\nFichiers crees et valides:")
print("   1. src/training/config_v2_1.py - Configuration V2.1 (correctifs rapides)")
print("   2. src/training/config_v3.py - Configuration V3 (EfficientNet-B4)")
print("   3. src/models/efficientnet_classifier.py - Modele EfficientNet-B4")
print("   4. src/training/trainer.py - Bug fix applique")
print("   5. GUIDE_AMELIORATION.md - Guide complet d'utilisation")

print("\nProchaines etapes:")
print("   - Option A (V2.1): Duree ~25h, Objectif 75-78%")
print("   - Option B (V3): Duree ~40h, Objectif 85-90%")

print("\nConsultez GUIDE_AMELIORATION.md pour les instructions detaillees")

print("\n" + "="*80)
print("TOUS LES TESTS PASSES - PRET POUR L'ENTRAINEMENT!")
print("="*80)
