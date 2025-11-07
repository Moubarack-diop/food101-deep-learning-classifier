"""
Script d'entraînement avec stratégie 2 phases
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.resnet_classifier import create_model
from src.data.dataset import get_dataloaders
from src.data.transforms import get_transforms, mixup_data, cutmix_data, mixup_criterion
from src.utils.metrics import calculate_accuracy, evaluate_model, print_metrics, AverageMeter
from src.training.config import Config
import random


class Trainer:
    """
    Classe pour gérer l'entraînement du modèle Food-101
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

        # Initialiser la phase avant _setup_training
        self.current_phase = 1
        self.best_acc = 0.0
        self.epochs_without_improvement = 0

        # Initialisation
        self._set_seed()
        self._setup_dataloaders()
        self._setup_model()
        self._setup_training()

        # Historique
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5_acc': []
        }

        print(f"\n Trainer initialisé sur {self.device}")

    def _set_seed(self):
        """Fixe la seed pour la reproductibilité"""
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.SEED)

    def _setup_dataloaders(self):
        """Configure les dataloaders"""
        print("\n Chargement des données...")

        # Transformations
        train_transform, test_transform = get_transforms(
            img_size=self.config.IMG_SIZE,
            augmentation_level=self.config.AUGMENTATION_LEVEL
        )

        # DataLoaders
        self.train_loader, self.test_loader, self.num_classes = get_dataloaders(
            root_dir=str(self.config.DATA_DIR),
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            train_transform=train_transform,
            test_transform=test_transform,
            pin_memory=self.config.PIN_MEMORY
        )

    def _setup_model(self):
        """Configure le modèle"""
        print("\n  Construction du modèle...")

        self.model = create_model(
            num_classes=self.config.NUM_CLASSES,
            pretrained=self.config.PRETRAINED,
            dropout=self.config.DROPOUT,
            phase=1  # Commencer en Phase 1
        )

        self.model = self.model.to(self.device)

    def _setup_training(self):
        """Configure l'entraînement (optimizer, loss, etc.)"""
        # Loss avec Label Smoothing
        label_smoothing = getattr(self.config, 'LABEL_SMOOTHING', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if label_smoothing > 0:
            print(f"✓ Label Smoothing activé: {label_smoothing}")

        # Optimizer Phase 1
        self.optimizer = self._get_optimizer(phase=1)

        # Scheduler
        if self.config.USE_SCHEDULER:
            self.scheduler = self._get_scheduler()
        else:
            self.scheduler = None

        # Mixed Precision
        if self.config.USE_AMP:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _get_optimizer(self, phase):
        """Crée l'optimizer selon la phase"""
        if phase == 1:
            optimizer_name = self.config.PHASE1_OPTIMIZER
            lr = self.config.PHASE1_LR
            weight_decay = self.config.PHASE1_WEIGHT_DECAY
        else:
            optimizer_name = self.config.PHASE2_OPTIMIZER
            lr = self.config.PHASE2_LR
            weight_decay = self.config.PHASE2_WEIGHT_DECAY

        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.PHASE2_MOMENTUM,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} non supporté")

        return optimizer

    def _get_scheduler(self):
        """Crée le scheduler"""
        if self.config.SCHEDULER_TYPE == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.STEP_SIZE,
                gamma=self.config.GAMMA
            )
        elif self.config.SCHEDULER_TYPE == 'cosine':
            # Utiliser PHASE2_EPOCHS pour T_max si on est en Phase 2
            t_max = self.config.PHASE2_EPOCHS if self.current_phase == 2 else self.config.PHASE1_EPOCHS
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max
            )
        elif self.config.SCHEDULER_TYPE == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=2
            )
        else:
            return None

    def train_epoch(self, epoch):
        """Entraîne le modèle pour une epoch"""
        self.model.train()

        losses = AverageMeter()
        top1_accs = AverageMeter()
        top5_accs = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.get_total_epochs()}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Appliquer MixUp ou CutMix si activé (seulement en Phase 2)
            use_mixup = False
            if self.current_phase == 2 and (self.config.USE_MIXUP or self.config.USE_CUTMIX):
                r = random.random()
                if r < self.config.MIXUP_PROB:
                    # Décider entre MixUp et CutMix
                    if self.config.USE_MIXUP and self.config.USE_CUTMIX:
                        if random.random() < 0.5:
                            images, labels_a, labels_b, lam = mixup_data(images, labels, self.config.MIXUP_ALPHA)
                        else:
                            images, labels_a, labels_b, lam = cutmix_data(images, labels, self.config.CUTMIX_ALPHA)
                    elif self.config.USE_MIXUP:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, self.config.MIXUP_ALPHA)
                    elif self.config.USE_CUTMIX:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, self.config.CUTMIX_ALPHA)
                    use_mixup = True

            # Forward avec AMP si activé
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    if use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)

                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.GRADIENT_CLIP:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.GRADIENT_CLIP:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)

                self.optimizer.step()

            # Métriques - FIX: Ne pas calculer accuracy si MixUp/CutMix appliqué
            # Car les labels sont mélangés et l'accuracy n'a pas de sens
            if use_mixup:
                # Avec MixUp/CutMix, on utilise seulement la loss pour optimisation
                # L'accuracy training ne sera pas calculée (artefact du mélange)
                top1, top5 = 0.0, 0.0
            else:
                # Accuracy normale seulement si pas de MixUp/CutMix
                top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            # Update accuracy meters seulement si pas de mixup
            if not use_mixup:
                top1_accs.update(top1, images.size(0))
                top5_accs.update(top5, images.size(0))

            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'top1': f'{top1_accs.avg:.2f}%',
                'top5': f'{top5_accs.avg:.2f}%'
            })

        return losses.avg, top1_accs.avg

    def validate(self):
        """Évalue le modèle sur le test set"""
        metrics = evaluate_model(
            self.model,
            self.test_loader,
            device=self.device,
            topk=self.config.TOPK
        )
        return metrics

    def switch_to_phase2(self):
        """Passe en Phase 2: fine-tuning complet"""
        print("\n" + "="*60)
        print(" PASSAGE À LA PHASE 2: FINE-TUNING COMPLET")
        print("="*60)

        # Dégeler le backbone
        self.model.unfreeze_backbone()
        self.model.print_param_info()

        # Nouvel optimizer avec LR plus petit
        self.optimizer = self._get_optimizer(phase=2)

        # Nouveau scheduler
        if self.config.USE_SCHEDULER:
            self.scheduler = self._get_scheduler()

        self.current_phase = 2
        print("✓ Phase 2 configurée\n")

    def save_checkpoint(self, epoch, is_best=False):
        """Sauvegarde le modèle"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'config': self.config
        }

        # Checkpoint régulier
        checkpoint_path = self.config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Meilleur modèle
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f" Meilleur modèle sauvegardé: {best_path}")

    def train(self):
        """Boucle d'entraînement complète (2 phases)"""
        self.config.print_config()

        print("\n DÉMARRAGE DE L'ENTRAÎNEMENT")
        print("="*60)

        start_time = time.time()

        # PHASE 1: Entraînement de la tête uniquement
        print("\n PHASE 1: Entraînement de la tête (backbone gelé)")
        print("-"*60)

        for epoch in range(1, self.config.PHASE1_EPOCHS + 1):
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate()

            # Sauvegarder l'historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['top1_acc'])
            self.history['val_top5_acc'].append(val_metrics['top5_acc'])

            # Afficher les métriques
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print_metrics(val_metrics, prefix="Validation")

            # Scheduler
            if self.scheduler and self.config.SCHEDULER_TYPE != 'plateau':
                self.scheduler.step()

            # Sauvegarder
            is_best = val_metrics['top1_acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['top1_acc']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

        # PHASE 2: Fine-tuning complet
        self.switch_to_phase2()

        print("\n PHASE 2: Fine-tuning complet")
        print("-"*60)

        for epoch in range(self.config.PHASE1_EPOCHS + 1, self.config.get_total_epochs() + 1):
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate()

            # Sauvegarder l'historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['top1_acc'])
            self.history['val_top5_acc'].append(val_metrics['top5_acc'])

            # Afficher les métriques
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print_metrics(val_metrics, prefix="Validation")

            # Scheduler
            if self.scheduler:
                if self.config.SCHEDULER_TYPE == 'plateau':
                    self.scheduler.step(val_metrics['top1_acc'])
                else:
                    self.scheduler.step()

            # Sauvegarder
            is_best = val_metrics['top1_acc'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['top1_acc']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping après {epoch} epochs (pas d'amélioration pendant {self.config.EARLY_STOPPING_PATIENCE} epochs)")
                break

        # Fin de l'entraînement
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(" ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        print(f"  Temps total: {total_time/3600:.2f} heures")
        print(f" Meilleure accuracy: {self.best_acc:.2f}%")
        print("="*60)

        return self.history


if __name__ == "__main__":
    # Créer le trainer et lancer l'entraînement
    trainer = Trainer(config=Config)
    history = trainer.train()

    # Sauvegarder l'historique
    import json
    history_path = Config.RESULTS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Historique sauvegardé: {history_path}")
