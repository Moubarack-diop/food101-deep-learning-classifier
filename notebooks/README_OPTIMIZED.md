# 📓 Notebook d'entraînement optimisé - Food-101 Classifier

## 📝 Fichier : `food101_training_optimized.ipynb`

Ce notebook est la version **optimisée et à jour** pour l'entraînement du modèle Food-101. Il remplace l'ancien notebook `food101_training_colab (1).ipynb`.

---

## ✨ Nouveautés et améliorations

### 🎯 Optimisations d'entraînement

| Paramètre | Ancien | Nouveau | Raison |
|-----------|--------|---------|--------|
| **Epochs Phase 1** | 3 | 5 | Meilleure adaptation de la tête |
| **Epochs Phase 2** | 10 | 30 | Convergence complète |
| **Learning Rate P2** | 1e-4 | 5e-5 | Fine-tuning plus fin |
| **Dropout** | 0.5 | 0.3 | Moins de régularisation excessive |
| **Augmentation** | medium | heavy | Dataset "in the wild" |
| **Scheduler** | step | cosine | Décroissance smooth du LR |
| **Early stopping** | 3 | 7 | Plus de patience |

### 🆕 Nouvelles fonctionnalités

- ✅ **MixUp augmentation** (α=0.2) pour meilleure généralisation
- ✅ **CutMix augmentation** (α=1.0) pour robustesse
- ✅ **Cosine Annealing Scheduler** adaptatif
- ✅ **Visualisations améliorées** (4 graphiques détaillés)
- ✅ **Sauvegarde automatique** sur Google Drive
- ✅ **Analyse détaillée** des résultats
- ✅ **Documentation complète** en français

### 📊 Performance attendue

- **Ancien modèle** : 63.36% Top-1 Accuracy
- **Nouveau modèle** : 85-88% Top-1 Accuracy (objectif)
- **Amélioration** : +22-25 points

---

## 🚀 Comment utiliser ce notebook

### 1️⃣ Ouvrir dans Google Colab

**Option A : Depuis Google Drive**
1. Ouvrez Google Drive
2. Naviguez vers `My Drive/deep_learning_project/notebooks/`
3. Double-cliquez sur `food101_training_optimized.ipynb`
4. Cliquez sur "Ouvrir avec Google Colaboratory"

**Option B : Directement sur Colab**
1. Allez sur [Google Colab](https://colab.research.google.com/)
2. `File` → `Upload notebook`
3. Sélectionnez `food101_training_optimized.ipynb`

### 2️⃣ Activer le GPU

⚠️ **IMPORTANT** : Le GPU est obligatoire pour l'entraînement

1. `Runtime` → `Change runtime type`
2. Sélectionnez **T4 GPU** (gratuit)
3. Cliquez sur `Save`

### 3️⃣ Exécuter le notebook

**Option A : Exécution automatique complète**
```
Runtime → Run all
```
Durée : 8-10 heures

**Option B : Exécution cellule par cellule**
```
Shift + Enter pour chaque cellule
```
Permet de vérifier chaque étape

### 4️⃣ Surveiller l'entraînement

Le notebook affichera :
- Configuration complète
- Progression de chaque epoch avec barre
- Métriques en temps réel (loss, accuracy)
- Graphiques de performance
- Sauvegarde automatique

---

## 📊 Structure du notebook

| Cellule | Description | Temps |
|---------|-------------|-------|
| 1 | Configuration GPU | 5s |
| 2 | Installation dépendances | 30s |
| 3 | Téléchargement dataset (5GB) | 5-10min |
| 4 | Import code depuis Drive | 10s |
| 5 | Affichage configuration | 5s |
| 6 | Test chargement données | 30s |
| 7 | **Entraînement (35 epochs)** | **8-10h** |
| 8 | Visualisation résultats | 30s |
| 9 | Métriques finales | 10s |
| 10 | Évaluation complète | 2-3min |
| 11 | Sauvegarde et téléchargement | 1-2min |

**Temps total estimé** : 8-10 heures

---

## 📁 Fichiers générés

Après l'entraînement, vous aurez :

```
checkpoints/
  └── best_model.pth           # Meilleur modèle (63.36% → 85-88%)

results/
  ├── training_history.json    # Historique complet epoch par epoch
  ├── training_summary.json    # Résumé avec config et résultats
  ├── final_metrics.json       # Métriques détaillées
  └── training_curves_optimized.png  # Graphiques (4 plots)

Archives téléchargées :
  ├── food101_results_YYYYMMDD_HHMMSS.zip        # Checkpoints
  └── food101_results_YYYYMMDD_HHMMSS_results.zip # Résultats

Google Drive backup :
  └── My Drive/deep_learning_project/results_backup/
```

---

## 🎯 Résultats attendus

### Performance finale

```json
{
  "top1_accuracy": "85-88%",
  "top5_accuracy": "97-99%",
  "improvement_vs_baseline": "+34-37%",
  "training_time": "8-10 hours on T4 GPU"
}
```

### Comparaison avec versions précédentes

| Version | Top-1 Acc | Top-5 Acc | Epochs | Temps |
|---------|-----------|-----------|--------|-------|
| Paper 2014 | 50.76% | ~80% | - | - |
| V1 (ancien notebook) | 63.36% | 86.68% | 10 | 2.5h |
| **V2 (ce notebook)** | **85-88%** | **97-99%** | **35** | **8-10h** |

---

## 🔧 Personnalisation

### Modifier les hyperparamètres

Après la cellule 5, ajoutez :

```python
# Exemple : réduire le nombre d'epochs pour test rapide
Config.PHASE1_EPOCHS = 2
Config.PHASE2_EPOCHS = 5

# Désactiver MixUp/CutMix
Config.USE_MIXUP = False
Config.USE_CUTMIX = False

# Changer le batch size (si GPU le permet)
Config.BATCH_SIZE = 64
```

### Mode debug rapide

Pour tester rapidement (1-2 heures) :

```python
from src.training.config import DebugConfig
trainer = Trainer(config=DebugConfig)
```

---

## ⚠️ Problèmes fréquents

### 1. GPU non disponible

```
⚠️ WARNING: GPU non disponible!
```

**Solution** :
- `Runtime` → `Change runtime type` → Sélectionner T4 GPU
- Redémarrer le runtime : `Runtime` → `Restart runtime`

### 2. Session Colab expirée

Si votre session Colab expire après 12h :

```python
# Reprendre l'entraînement depuis le dernier checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_X.pth')
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Relancer l'entraînement
```

### 3. Erreur d'importation du code

```
ModuleNotFoundError: No module named 'src'
```

**Solution** : Vérifiez que le code est bien copié depuis Drive (Cellule 4)

### 4. Mémoire GPU insuffisante

```
RuntimeError: CUDA out of memory
```

**Solution** : Réduire le batch size
```python
Config.BATCH_SIZE = 16  # Au lieu de 32
```

---

## 💡 Conseils d'utilisation

### ✅ Bonnes pratiques

1. **Vérifier le GPU** : Toujours vérifier cellule 1 que le GPU est activé
2. **Surveiller l'entraînement** : Ne pas fermer l'onglet pendant l'entraînement
3. **Colab Pro** : Pour sessions plus longues (24h au lieu de 12h)
4. **Sauvegardes** : Le notebook sauvegarde automatiquement toutes les 5 epochs

### 🚫 À éviter

1. ❌ Fermer l'onglet pendant l'entraînement
2. ❌ Oublier d'activer le GPU
3. ❌ Modifier le code pendant l'exécution
4. ❌ Utiliser CPU (100x plus lent)

---

## 📈 Suivi de l'entraînement

### Métriques à surveiller

**Phase 1 (epochs 1-5)** :
- Validation accuracy devrait atteindre ~55-60%
- Loss devrait descendre de ~3.0 à ~1.7

**Phase 2 (epochs 6-35)** :
- Validation accuracy devrait monter progressivement
- Early stopping activé si pas d'amélioration pendant 7 epochs
- Meilleure accuracy attendue vers epoch 25-30

### Indicateurs de bon entraînement

✅ Loss qui descend régulièrement
✅ Validation accuracy qui monte
✅ Écart Train-Val stable (<5%)
✅ Top-5 accuracy >95%

### Indicateurs de problèmes

⚠️ Loss qui remonte (overfitting)
⚠️ Validation accuracy qui stagne tôt
⚠️ Écart Train-Val qui augmente (>15%)
⚠️ Loss qui ne descend plus

---

## 📞 Support

Si vous rencontrez des problèmes :

1. **Vérifier les logs** : Lire attentivement les messages d'erreur
2. **Redémarrer le runtime** : `Runtime` → `Restart runtime`
3. **Vérifier les fichiers** : S'assurer que `src/` est bien copié
4. **Mode debug** : Utiliser `DebugConfig` pour test rapide

---

## 🎓 Pour aller plus loin

### Améliorer encore les performances

Si 85-88% n'est pas atteint :

1. **Augmenter epochs** : 50 au lieu de 30 en Phase 2
2. **Tester EfficientNet-B4** : Souvent meilleur que ResNet-50
3. **Label Smoothing** : Ajouter (α=0.1)
4. **Test-Time Augmentation** : Multi-crop evaluation
5. **Ensemble de modèles** : Moyenner plusieurs modèles

### Architecture alternative

```python
# Dans config.py, ajouter :
MODEL_NAME = 'efficientnet_b4'  # Au lieu de resnet50

# Gain attendu : +2-5% accuracy
```

---

## 📚 Références

**Papers** :
- Food-101 dataset : Bossard et al., ECCV 2014
- MixUp : Zhang et al., ICLR 2018
- CutMix : Yun et al., ICCV 2019
- ResNet : He et al., CVPR 2016

**Code** :
- PyTorch : https://pytorch.org/
- Timm : https://github.com/huggingface/pytorch-image-models

---

**Dernière mise à jour** : Janvier 2025
**Version** : 2.0 (Optimisée)
**Auteur** : Mouhamed Diop | DIC2-GIT

---

🎯 **Objectif** : 85-88% Top-1 Accuracy
⏱️ **Temps** : 8-10 heures
🚀 **Prêt à entraîner !**
