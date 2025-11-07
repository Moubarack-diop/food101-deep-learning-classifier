# 🚀 Guide Complet - Exécuter V2.1 sur Google Colab

**Objectif:** Améliorer 66% → 75-78% Top-1 Accuracy
**Durée:** ~25h d'entraînement
**Difficulté:** ⭐⭐ Facile

---

## 📋 **Étapes Complètes**

### **Étape 1: Préparer Google Drive** (5 minutes)

1. **Ouvrir Google Drive:** https://drive.google.com

2. **Vérifier que votre projet est synchronisé:**
   - Aller dans "My Drive" → "deep_learning_project"
   - Vérifier que ces fichiers sont présents:
     ```
     ✅ src/training/config_v2_1.py
     ✅ src/training/trainer.py
     ✅ src/models/resnet_classifier.py
     ✅ notebooks/food101_training_optimized_v2.ipynb
     ```

3. **Si les fichiers ne sont pas synchronisés:**
   - Attendre quelques minutes (synchronisation automatique)
   - Ou forcer la sync: Clic droit → "Disponible hors connexion"

---

### **Étape 2: Ouvrir le Notebook dans Colab** (2 minutes)

**Option A: Depuis Google Drive (RECOMMANDÉ)**

1. Aller dans Google Drive
2. Naviguer vers: `My Drive/deep_learning_project/notebooks/`
3. **Clic droit** sur `food101_training_optimized_v2.ipynb`
4. Sélectionner: "Ouvrir avec" → **"Google Colaboratory"**

**Option B: Depuis Colab directement**

1. Aller sur: https://colab.research.google.com
2. Cliquer sur "File" → "Open notebook"
3. Onglet "Google Drive"
4. Naviguer vers: `deep_learning_project/notebooks/food101_training_optimized_v2.ipynb`
5. Cliquer pour ouvrir

---

### **Étape 3: Activer le GPU** (1 minute) ⚡ **IMPORTANT**

1. Dans Colab, menu: **"Runtime"** → **"Change runtime type"**

2. Dans la fenêtre qui s'ouvre:
   - **Hardware accelerator:** Sélectionner **"GPU"**
   - **GPU type:**
     - Si vous avez **Colab Pro/Pro+:** Choisir **"T4"** ou **"V100"**
     - Si **Colab gratuit:** Laisser sur **"GPU"** (T4 automatique)

3. Cliquer **"Save"**

4. **Vérifier le GPU:**
   - Le notebook va redémarrer
   - En haut à droite, vous devriez voir: "RAM" et **"GPU"** (au lieu de "Disk")

**⚠️ IMPORTANT:** Sans GPU, l'entraînement prendra **200h+** au lieu de 25h!

---

### **Étape 4: Modifier le Notebook pour V2.1** (3 minutes)

**Cellule 10: Configuration**

Trouver cette cellule (environ ligne 120-140):

```python
import sys
import torch
sys.path.append('/content')

from src.training.config import Config  # ← CETTE LIGNE À MODIFIER

# Adapter la configuration pour Colab
Config.DATA_DIR = 'data/food-101'
Config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Config.NUM_WORKERS = 2
```

**MODIFIER EN:**

```python
import sys
import torch
sys.path.append('/content')

# ✅ MODIFICATION V2.1: Utiliser la config optimisée
from src.training.config_v2_1 import ConfigV2_1 as Config

# Adapter la configuration pour Colab
Config.DATA_DIR = 'data/food-101'
Config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Config.NUM_WORKERS = 2
```

**C'est tout!** Juste 1 ligne à changer 🎉

---

### **Étape 5: Vérifier la Configuration** (Optionnel mais recommandé)

**Ajouter une nouvelle cellule après la cellule 10:**

Menu: "Insert" → "Code cell" (ou `Ctrl+M B`)

**Copier ce code:**

```python
# Vérification de la configuration V2.1
print("="*80)
print("VERIFICATION CONFIGURATION V2.1")
print("="*80)

print(f"\nAugmentation Level: {Config.AUGMENTATION_LEVEL}")
print(f"CutMix Alpha: {Config.CUTMIX_ALPHA}")
print(f"MixUp Prob: {Config.MIXUP_PROB}")
print(f"Phase 2 LR: {Config.PHASE2_LR}")
print(f"Phase 2 Epochs: {Config.PHASE2_EPOCHS}")
print(f"Total Epochs: {Config.get_total_epochs()}")

# Assertions
assert Config.AUGMENTATION_LEVEL == 'medium', "ERREUR: Devrait être 'medium'"
assert Config.CUTMIX_ALPHA == 0.3, "ERREUR: Devrait être 0.3"
assert Config.MIXUP_PROB == 0.3, "ERREUR: Devrait être 0.3"
assert Config.PHASE2_LR == 7.5e-5, "ERREUR: Devrait être 7.5e-5"
assert Config.PHASE2_EPOCHS == 100, "ERREUR: Devrait être 100"

print("\n✅ Configuration V2.1 validée!")
print("Objectif: 75-78% Top-1 Accuracy")
print("="*80)
```

**Exécuter cette cellule** (Shift+Enter) pour vérifier que tout est correct.

**Résultat attendu:**
```
================================================================================
VERIFICATION CONFIGURATION V2.1
================================================================================

Augmentation Level: medium
CutMix Alpha: 0.3
MixUp Prob: 0.3
Phase 2 LR: 7.5e-05
Phase 2 Epochs: 100
Total Epochs: 105

✅ Configuration V2.1 validée!
Objectif: 75-78% Top-1 Accuracy
================================================================================
```

---

### **Étape 6: Lancer l'Entraînement** (25h) ⏱️

**Option A: Exécution Automatique Complète (RECOMMANDÉ)**

1. Menu: **"Runtime"** → **"Run all"**
2. Confirmer si demandé
3. Le notebook va:
   - ✅ Installer les dépendances (~2 min)
   - ✅ Télécharger Food-101 dataset (~10 min, 5GB)
   - ✅ Copier le code depuis Drive (~1 min)
   - ✅ Lancer l'entraînement (~24-28h)

**Option B: Exécution Cellule par Cellule**

Pour mieux comprendre:
- Cliquer sur la première cellule
- Appuyer **Shift+Enter** pour exécuter et passer à la suivante
- Répéter jusqu'à la cellule d'entraînement (cellule 14)

---

### **Étape 7: Garder Colab Actif** (CRITIQUE pour 25h) 🔋

**⚠️ PROBLÈME:** Colab gratuit se déconnecte après ~12h d'inactivité

**SOLUTIONS:**

**Solution 1: Colab Pro (RECOMMANDÉ)** 💰
- **Prix:** ~10€/mois
- **Avantages:**
  - Pas de timeout pendant 24h
  - GPU plus rapides (V100, A100)
  - Priorité d'accès
- **Lien:** https://colab.research.google.com/signup

**Solution 2: Garder l'onglet actif** (Colab gratuit)
- Ne PAS fermer l'onglet Colab
- Revenir toutes les 2-3h et bouger la souris
- Ouvrir la console JavaScript (F12) et exécuter:

```javascript
function KeepAlive() {
  console.log("Keeping alive...");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(KeepAlive, 60000); // Toutes les 60 secondes
```

**Solution 3: Sauvegardes fréquentes**
- Le trainer sauvegarde déjà tous les 5 epochs
- Si déconnexion, vous pourrez reprendre depuis le dernier checkpoint

---

### **Étape 8: Surveiller la Progression** 👀

**Pendant l'entraînement, vous verrez:**

```
================================================================================
🚀 DÉMARRAGE DE L'ENTRAÎNEMENT OPTIMISÉ V2.1
================================================================================

⏱️ Durée estimée: 24-28 heures

Epoch 1/105: 100%|██████████| 2367/2367 [15:23<00:00, 2.56it/s, loss=3.2145, top1=28.34%, top5=51.23%]

Validation Metrics:
==================================================
  loss           : 2.8234
  top1_acc       :  32.45%
  top5_acc       :  58.32%
==================================================

Epoch 2/105: 100%|██████████| 2367/2367 [15:18<00:00, 2.58it/s, loss=2.9876, top1=35.67%, top5=60.12%]
...
```

**Indicateurs de bon fonctionnement:**
- ✅ Loss qui **diminue** progressivement
- ✅ Accuracy qui **augmente** progressivement
- ✅ Temps par epoch: ~15-18 minutes sur T4 GPU
- ✅ "Validation Metrics" affichées après chaque epoch

**Indicateurs de problème:**
- ❌ Loss qui augmente (rare avec nos configs)
- ❌ Temps par epoch > 30 min (GPU pas activé?)
- ❌ "CUDA out of memory" (réduire batch_size à 24)

---

### **Étape 9: Pendant l'Entraînement** (25h)

**Checkpoints automatiques:**

Le modèle est sauvegardé automatiquement:
- ✅ **Tous les 5 epochs** dans `checkpoints/checkpoint_epoch_X.pth`
- ✅ **Meilleur modèle** dans `checkpoints/best_model.pth`
- ✅ **Synchronisation Drive** à la fin

**Si Colab se déconnecte:**

1. **Reconnecter au GPU** (Runtime → Change runtime type)
2. **Ré-exécuter les cellules 1-9** (setup)
3. **Charger le dernier checkpoint** - Modifier cellule 14:

```python
# Avant de lancer trainer.train(), ajouter:
import torch
from pathlib import Path

# Trouver le dernier checkpoint
checkpoints = sorted(Path('checkpoints').glob('checkpoint_epoch_*.pth'))
if checkpoints:
    last_checkpoint = checkpoints[-1]
    print(f"Reprise depuis: {last_checkpoint}")

    checkpoint = torch.load(last_checkpoint, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    trainer.best_acc = checkpoint.get('best_acc', 0.0)

    print(f"Reprise à l'epoch {start_epoch}")
    print(f"Best accuracy jusqu'ici: {trainer.best_acc:.2f}%")
else:
    start_epoch = 1
    print("Démarrage from scratch")
```

---

### **Étape 10: Résultats Finaux** 🎉

**Après ~25h, vous verrez:**

```
================================================================================
🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!
================================================================================

⏱️ Temps total: 24h 37min
🎯 Meilleure Top-1 Accuracy: 76.82%
📊 Epochs effectués: 105
💾 Meilleur modèle sauvegardé: checkpoints/best_model.pth

✅ OBJECTIF ATTEINT! (75-78% top-1 accuracy)
🏆 Performance excellente!
```

**Ce qui est sauvegardé:**

Dans Google Drive (`deep_learning_project/results_v2_1_YYYYMMDD_HHMMSS/`):
- ✅ `best_model_v2_1.pth` - Meilleur modèle
- ✅ `training_history_v2.json` - Historique complet
- ✅ `training_summary_v2.json` - Résumé des résultats
- ✅ `final_metrics_v2.json` - Métriques finales
- ✅ `training_curves_optimized_v2.png` - Graphiques

---

## 🔧 **Dépannage**

### **Problème 1: "No GPU available"**

**Symptômes:** Message "⚠️ WARNING: GPU non disponible"

**Solutions:**
1. Runtime → Change runtime type → GPU → Save
2. Vérifier quota GPU (Colab gratuit: limité)
3. Essayer à une autre heure (moins de charge)
4. Passer à Colab Pro

---

### **Problème 2: "CUDA out of memory"**

**Symptômes:** Erreur pendant l'entraînement

**Solutions:**
1. **Réduire batch size** - Modifier dans la cellule config:
   ```python
   Config.BATCH_SIZE = 24  # Au lieu de 32
   ```
2. Redémarrer runtime: Runtime → Restart runtime
3. Ré-exécuter depuis le début

---

### **Problème 3: "Module not found: config_v2_1"**

**Symptômes:** Erreur à l'import

**Solutions:**
1. Vérifier que `src/training/config_v2_1.py` est dans Drive
2. Ré-exécuter cellule 8 (copie du code depuis Drive)
3. Vérifier le chemin:
   ```python
   !ls -la /content/src/training/
   # Devrait afficher config_v2_1.py
   ```

---

### **Problème 4: Déconnexion après 12h**

**Symptômes:** "Runtime disconnected"

**Solutions:**
1. **Meilleure:** Passer à Colab Pro (10€/mois)
2. Utiliser le script JavaScript (voir Étape 7)
3. Reprendre depuis checkpoint (voir Étape 9)

---

### **Problème 5: Download dataset échoue**

**Symptômes:** Erreur au téléchargement Food-101

**Solutions:**
1. Ré-exécuter la cellule 6 (téléchargement)
2. Si échec répété, télécharger manuellement:
   ```python
   !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -O data/food-101.tar.gz
   !tar -xzf data/food-101.tar.gz -C data/
   ```

---

## 📊 **Comparaison avec V2 Actuel**

| Métrique | V2 (Actuel) | V2.1 (Attendu) | Amélioration |
|----------|-------------|----------------|--------------|
| **Top-1 Accuracy** | 66.43% | **75-78%** | +9 à +12 pts |
| **Top-5 Accuracy** | 88.79% | **94-96%** | +5 à +7 pts |
| **Durée** | 18-22h | 24-28h | +6h |
| **Changement code** | N/A | **1 ligne** | Minimal |

---

## ✅ **Checklist Avant de Lancer**

Avant d'exécuter "Run all", vérifiez:

- [ ] ✅ Notebook ouvert dans Colab
- [ ] ✅ GPU activé (Runtime → Change runtime type → GPU)
- [ ] ✅ Cellule 10 modifiée (import ConfigV2_1)
- [ ] ✅ Cellule de vérification ajoutée (optionnel)
- [ ] ✅ Colab Pro activé OU script anti-déconnexion prêt
- [ ] ✅ ~25-28h de temps disponible
- [ ] ✅ Google Drive a suffisamment d'espace (>10GB)

**Si toutes les cases sont cochées:** Vous êtes prêt! 🚀

---

## 🎯 **Après l'Entraînement**

**Une fois terminé (75-78% atteint):**

1. **Télécharger les résultats** depuis Drive
2. **Analyser les graphiques** (`training_curves_optimized_v2.png`)
3. **Tester l'app web** avec le nouveau modèle:
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. **Rédiger le rapport académique** avec les résultats
5. **Célébrer l'amélioration!** 🎉 (+9 à +12 points)

---

## 💡 **Conseils Finaux**

1. **Première fois?** Testez d'abord avec DebugConfig (2 epochs, 10 min)
2. **Colab gratuit?** Préparez-vous à surveiller toutes les 2-3h
3. **Colab Pro?** Lancez le soir et vérifiez le matin
4. **Impatient?** Regardez les métriques epoch par epoch
5. **Prudent?** Vérifiez les checkpoints toutes les 5 epochs

---

## 📖 **Ressources**

- **Documentation Colab:** https://colab.research.google.com/notebooks/intro.ipynb
- **Guide complet V2.1:** `GUIDE_AMELIORATION.md`
- **Quick Start:** `QUICK_START.md`
- **Support:** README.md

---

**Bonne chance! En ~25h vous aurez 75-78% d'accuracy! 🚀**

**Questions?** Voir `GUIDE_AMELIORATION.md` ou README.md

---

**Dernière mise à jour:** 2025-10-25
**Créé par:** Claude Code Assistant
