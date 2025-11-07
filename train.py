"""
Script de lancement rapide pour l'entraînement
"""

import argparse
from src.training.trainer import Trainer
from src.training.config import Config, DebugConfig


def main():
    parser = argparse.ArgumentParser(description="Entraîner le modèle Food-101")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Mode debug (moins d\'epochs)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Chemin vers le dossier food-101'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Taille des batchs'
    )
    parser.add_argument(
        '--phase1-epochs',
        type=int,
        default=None,
        help='Nombre d\'epochs Phase 1'
    )
    parser.add_argument(
        '--phase2-epochs',
        type=int,
        default=None,
        help='Nombre d\'epochs Phase 2'
    )
    parser.add_argument(
        '--lr1',
        type=float,
        default=None,
        help='Learning rate Phase 1'
    )
    parser.add_argument(
        '--lr2',
        type=float,
        default=None,
        help='Learning rate Phase 2'
    )

    args = parser.parse_args()

    # Choisir la config
    if args.debug:
        config = DebugConfig
        print("🔧 Mode DEBUG activé")
    else:
        config = Config

    # Overrides
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.phase1_epochs:
        config.PHASE1_EPOCHS = args.phase1_epochs
    if args.phase2_epochs:
        config.PHASE2_EPOCHS = args.phase2_epochs
    if args.lr1:
        config.PHASE1_LR = args.lr1
    if args.lr2:
        config.PHASE2_LR = args.lr2

    # Créer le trainer
    trainer = Trainer(config=config)

    # Lancer l'entraînement
    history = trainer.train()

    print("\n Entraînement terminé!")
    print(f"Meilleure accuracy: {trainer.best_acc:.2f}%")


if __name__ == "__main__":
    main()
