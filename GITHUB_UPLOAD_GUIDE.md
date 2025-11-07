# 📤 Guide pour Publier sur GitHub

## 🎯 Étapes Complètes

### **Étape 1 : Initialiser Git Localement**

```bash
# Aller dans le dossier du projet
cd "D:\My Drive\deep_learning_project"

# Initialiser git
git init

# Vérifier que .gitignore existe
ls -la .gitignore
```

### **Étape 2 : Configurer Git (si pas déjà fait)**

```bash
# Configurer votre nom (remplacer par votre nom)
git config --global user.name "Mouhamed Diop"

# Configurer votre email GitHub
git config --global user.email "votre.email@example.com"

# Vérifier la configuration
git config --list
```

### **Étape 3 : Ajouter les Fichiers**

```bash
# Ajouter tous les fichiers (le .gitignore exclut automatiquement les gros fichiers)
git add .

# Vérifier ce qui va être commité
git status

# Vous devriez voir :
# - src/ (ajouté)
# - app/ (ajouté)
# - notebooks/ (ajouté)
# - train.py (ajouté)
# - requirements.txt (ajouté)
# - README_GITHUB.md (ajouté)
# - etc.

# Vous NE devriez PAS voir :
# - data/food-101/images/ (ignoré par .gitignore)
# - checkpoints/ (ignoré)
# - *.pth (ignoré)
# - food-101.tar.gz (ignoré)
```

### **Étape 4 : Premier Commit**

```bash
# Créer le commit initial
git commit -m "Initial commit: Food-101 Classification Project

- Architecture ResNet-50 et EfficientNet-B4
- Entraînement 2 phases avec Transfer Learning
- Augmentation avancée: MixUp, CutMix, Random Erasing
- Application web Streamlit
- Documentation complète
- Résultats: 87.21% Top-1 Accuracy (V3)"
```

### **Étape 5 : Créer le Repository sur GitHub**

1. **Aller sur GitHub** : https://github.com/Moubarack-diop
2. **Cliquer sur "New" (ou "New repository")**
3. **Remplir les informations :**
   - **Repository name :** `food101-deep-learning-classifier`
   - **Description :** `🍕 Automatic food image classification using Deep Learning and Transfer Learning on Food-101 dataset. Achieves 87.21% Top-1 accuracy with EfficientNet-B4.`
   - **Visibility :** Public (ou Private si vous préférez)
   - **⚠️ Ne PAS cocher** "Initialize with README" (on a déjà un README)
   - **⚠️ Ne PAS ajouter** .gitignore (on en a déjà un)
   - **⚠️ Ne PAS ajouter** License (on en a déjà un)
4. **Cliquer sur "Create repository"**

### **Étape 6 : Lier le Repository Local au Remote**

GitHub va vous montrer des commandes. Utilisez celles-ci :

```bash
# Ajouter le remote (remplacer par votre URL)
git remote add origin https://github.com/Moubarack-diop/food101-deep-learning-classifier.git

# Vérifier que le remote est ajouté
git remote -v
```

### **Étape 7 : Pousser vers GitHub**

```bash
# Renommer la branche en 'main' (standard GitHub)
git branch -M main

# Pousser vers GitHub
git push -u origin main
```

**⏳ Attendre que l'upload se termine** (peut prendre quelques minutes selon la taille)

---

## ✅ **Vérifications Après Upload**

1. **Aller sur** : https://github.com/Moubarack-diop/food101-deep-learning-classifier
2. **Vérifier que vous voyez :**
   - ✅ README.md affiché avec badges et tableaux
   - ✅ Dossiers : src/, app/, notebooks/, data/
   - ✅ Fichiers : train.py, requirements.txt, LICENSE
   - ❌ **PAS de** : checkpoints/, food-101.tar.gz, *.pth

3. **Tester les badges** (peuvent mettre quelques minutes à s'afficher)

---

## 🎨 **Personnalisations Recommandées**

### **1. Ajouter une Image de Démo**

Créer un dossier `assets/` avec des screenshots :

```bash
mkdir assets
# Copier des captures d'écran de votre app Streamlit
cp screenshot.png assets/
git add assets/
git commit -m "Add demo screenshot"
git push
```

Puis dans README_GITHUB.md :
```markdown
## 📸 Aperçu

![Demo](assets/demo.png)
```

### **2. Ajouter des Topics sur GitHub**

Sur la page du repo GitHub :
1. Cliquer sur ⚙️ (Settings) en haut à droite
2. Chercher "Topics"
3. Ajouter : `deep-learning`, `pytorch`, `food-classification`, `computer-vision`, `transfer-learning`, `resnet`, `efficientnet`, `streamlit`, `food-101`

### **3. Créer un GitHub Pages (optionnel)**

Pour héberger la documentation :
1. Settings → Pages
2. Source : Deploy from branch `main`
3. Folder : `/docs`

### **4. Ajouter des GitHub Actions (optionnel)**

Pour tests automatiques :

Créer `.github/workflows/tests.yml` :
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/
```

---

## 🔄 **Commandes Git Utiles pour la Suite**

### **Ajouter de nouveaux changements**

```bash
# Voir les fichiers modifiés
git status

# Ajouter les modifications
git add .

# Ou ajouter des fichiers spécifiques
git add src/models/new_model.py

# Commit
git commit -m "Add new model architecture"

# Push
git push
```

### **Créer une nouvelle branche pour features**

```bash
# Créer et basculer sur nouvelle branche
git checkout -b feature/amelioration-v4

# Faire des modifications...

# Commit et push
git add .
git commit -m "Add V4 with Vision Transformer"
git push -u origin feature/amelioration-v4

# Ensuite créer une Pull Request sur GitHub
```

### **Mettre à jour depuis GitHub**

```bash
# Récupérer les derniers changements
git pull origin main
```

### **Voir l'historique**

```bash
# Historique des commits
git log --oneline --graph --all

# Différences
git diff
```

---

## 📊 **README.md Final à Afficher**

Une fois uploadé, **renommer** `README_GITHUB.md` en `README.md` :

```bash
# Localement
mv README.md README_OLD.md
mv README_GITHUB.md README.md

# Commit
git add .
git commit -m "Update README for GitHub"
git push
```

---

## 🐛 **Dépannage**

### **Problème : "fatal: not a git repository"**
```bash
git init
```

### **Problème : "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/Moubarack-diop/food101-deep-learning-classifier.git
```

### **Problème : Upload trop lent (> 100MB fichiers)**
Vérifier que `.gitignore` fonctionne :
```bash
git ls-files | grep -E "\.(pth|tar\.gz)$"
```
Si des gros fichiers apparaissent :
```bash
git rm --cached file.pth
git commit -m "Remove large file"
```

### **Problème : "Authentication failed"**
Utiliser un Personal Access Token au lieu du mot de passe :
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Utiliser le token comme mot de passe

### **Problème : Fichier > 100MB**
GitHub limite à 100MB par fichier. Si besoin :
1. Vérifier `.gitignore` inclut le fichier
2. Ou utiliser Git LFS (Large File Storage)

---

## ✨ **Commandes Rapides (Copy-Paste)**

```bash
# Tout en une fois
cd "D:\My Drive\deep_learning_project"
git init
git add .
git commit -m "Initial commit: Food-101 Deep Learning Classifier

- ResNet-50 & EfficientNet-B4 architectures
- 87.21% Top-1 accuracy achieved
- Complete documentation and Streamlit app"

git remote add origin https://github.com/Moubarack-diop/food101-deep-learning-classifier.git
git branch -M main
git push -u origin main
```

---

## 🎯 **Checklist Finale**

Avant de publier :

- [ ] `.gitignore` créé et vérifié
- [ ] README_GITHUB.md créé et complet
- [ ] LICENSE ajouté
- [ ] Données sensibles supprimées (emails, API keys)
- [ ] Gros fichiers exclus (dataset, checkpoints)
- [ ] Code commenté et propre
- [ ] Documentation à jour
- [ ] Repository GitHub créé
- [ ] Remote configuré
- [ ] Premier push réussi
- [ ] README s'affiche correctement sur GitHub
- [ ] Topics ajoutés

---

**Vous êtes prêt à partager votre projet avec le monde ! 🚀**

**Lien de votre repo :** https://github.com/Moubarack-diop/food101-deep-learning-classifier
