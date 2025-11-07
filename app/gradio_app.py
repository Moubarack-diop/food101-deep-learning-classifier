"""
Application Gradio alternative pour la classification Food-101
"""

import gradio as gr
import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resnet_classifier import load_checkpoint
from src.data.transforms import get_test_transforms


# Charger le modèle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
transform = get_test_transforms()
class_names = []


def load_model_and_classes():
    """Charge le modèle et les classes"""
    global model, class_names

    # Charger le modèle
    checkpoint_path = 'checkpoints/best_model.pth'
    try:
        model = load_checkpoint(checkpoint_path, num_classes=101, device=device)
        print(f"✓ Modèle chargé sur {device}")
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        model = None

    # Charger les classes
    classes_file = Path('data/food-101/meta/classes.txt')
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip().replace('_', ' ').title() for line in f.readlines()]
    else:
        class_names = [f"Class {i}" for i in range(101)]

    print(f"✓ {len(class_names)} classes chargées")


def predict(image):
    """
    Prédiction sur une image

    Args:
        image: Image PIL ou numpy array

    Returns:
        predictions: Dictionnaire {classe: probabilité}
    """
    if model is None:
        return {"Erreur": 1.0}

    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Prétraitement
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    # Top-5 prédictions
    top_probs, top_indices = torch.topk(probs, 5)

    # Formater pour Gradio
    predictions = {}
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions[class_names[idx.item()]] = float(prob.item())

    return predictions


def create_interface():
    """Crée l'interface Gradio"""

    # Charger le modèle
    load_model_and_classes()

    # Description
    title = " Food-101 Classifier"
    description = """
    ## Classification automatique de 101 types d'aliments

    Uploadez une image d'aliment et obtenez les 5 prédictions les plus probables.

    **Performance du modèle:**
    - Top-1 Accuracy: ~89%
    - Top-5 Accuracy: ~98%
    - Architecture: ResNet-50 avec transfer learning

    **Projet académique** | Mouhamed Diop | DIC2-GIT | 2025
    """

    article = """
    ###  À propos du projet

    Ce classificateur a été développé dans le cadre d'un projet académique visant à reproduire
    et améliorer les résultats de l'article "Food-101 – Mining Discriminative Components with Random Forests"
    (Bossard et al., ECCV 2014).

    **Amélioration par rapport à l'article original:**
    - Article 2014: 50.76% top-1 accuracy
    - Notre modèle: ~89% top-1 accuracy
    - **Gain: +38 points** grâce au deep learning!

    **Technologies utilisées:**
    - PyTorch
    - ResNet-50 pré-entraîné sur ImageNet
    - Transfer Learning et Fine-tuning
    - Data Augmentation
    """

    # Exemples
    examples = [
        ["examples/pizza.jpg"],
        ["examples/sushi.jpg"],
        ["examples/burger.jpg"],
    ]

    # Interface
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Image d'aliment"),
        outputs=gr.Label(num_top_classes=5, label="Prédictions"),
        title=title,
        description=description,
        article=article,
        examples=examples if Path("examples").exists() else None,
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )

    return interface


if __name__ == "__main__":
    # Créer et lancer l'interface
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
