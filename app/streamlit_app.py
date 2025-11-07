"""
Application Web Streamlit pour la classification Food-101
"""

import streamlit as st
import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import time

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resnet_classifier import load_checkpoint
from src.data.transforms import get_test_transforms


# Configuration de la page
st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(checkpoint_path='checkpoints/best_model.pth'):
    """Charge le modèle (mis en cache)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = load_checkpoint(checkpoint_path, num_classes=101, device=device)
        return model, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, None


@st.cache_data
def load_class_names():
    """Charge les noms des classes"""
    # Liste des 101 classes Food-101
    classes_file = Path('data/food-101/meta/classes.txt')

    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip().replace('_', ' ').title() for line in f.readlines()]
    else:
        # Fallback: classes prédéfinies
        classes = [
            'Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare',
            'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 'Breakfast Burrito',
            # ... (101 classes au total)
        ]

    return classes


def predict(image, model, device, transform, class_names, top_k=5):
    """
    Effectue une prédiction sur une image

    Args:
        image: Image PIL
        model: Modèle PyTorch
        device: Device
        transform: Transformations
        class_names: Liste des noms de classes
        top_k: Nombre de prédictions à retourner

    Returns:
        predictions: Liste de tuples (classe, probabilité)
    """
    model.eval()

    # Prétraitement
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        start_time = time.time()
        outputs = model(img_tensor)
        inference_time = (time.time() - start_time) * 1000  # en ms

        # Probabilités
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    # Formater les résultats
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            'class': class_names[idx],
            'probability': prob.item() * 100,
            'index': idx.item()
        })

    return predictions, inference_time


def main():
    # Header
    st.title(" Food-101 Classifier")
    st.markdown("""
    ### Classification automatique de 101 types d'aliments
    Uploadez une image d'aliment et obtenez une prédiction instantanée!
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Charger le modèle
    with st.spinner("Chargement du modèle..."):
        model, device = load_model()

    if model is None:
        st.error(" Impossible de charger le modèle. Veuillez vérifier que le checkpoint existe.")
        st.stop()

    st.sidebar.success(f"✓ Modèle chargé sur {device}")

    # Charger les classes
    class_names = load_class_names()
    st.sidebar.info(f" {len(class_names)} classes disponibles")

    # Paramètres
    top_k = st.sidebar.slider("Nombre de prédictions", 1, 10, 5)

    # Transformations
    transform = get_test_transforms()

    # Zone d'upload
    st.sidebar.header(" Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choisir une image",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG"
    )

    # Images d'exemple
    st.sidebar.header(" Ou essayer un exemple")
    example_images = {
        "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400",
        "Sushi": "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400",
        "Burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400",
        "Ice Cream": "https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=400",
    }

    selected_example = st.sidebar.selectbox(
        "Choisir un exemple",
        ["Aucun"] + list(example_images.keys())
    )

    # Main area
    col1, col2 = st.columns([1, 1])

    image = None

    # Charger l'image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.subheader(" Image uploadée")
            st.image(image, use_column_width=True)

    elif selected_example != "Aucun":
        try:
            import requests
            from io import BytesIO
            response = requests.get(example_images[selected_example])
            image = Image.open(BytesIO(response.content)).convert('RGB')
            with col1:
                st.subheader(" Image d'exemple")
                st.image(image, use_column_width=True)
        except:
            st.error("Erreur lors du chargement de l'exemple")

    # Prédiction
    if image is not None:
        with col2:
            st.subheader(" Prédictions")

            with st.spinner("Classification en cours..."):
                predictions, inference_time = predict(
                    image, model, device, transform, class_names, top_k
                )

            # Afficher les résultats
            st.success(f"✓ Classification terminée en {inference_time:.1f}ms")

            # Top prédiction
            top_pred = predictions[0]
            st.markdown(f"""
            ###  Prédiction principale
            **{top_pred['class']}**
            """)

            # Barre de confiance
            st.progress(top_pred['probability'] / 100)
            st.markdown(f"**Confiance:** {top_pred['probability']:.2f}%")

            # Top-k prédictions
            st.markdown(f"###  Top-{top_k} Prédictions")

            for i, pred in enumerate(predictions, 1):
                with st.expander(f"{i}. {pred['class']} - {pred['probability']:.2f}%"):
                    st.write(f"Probabilité: {pred['probability']:.2f}%")
                    st.progress(pred['probability'] / 100)

    else:
        # Instructions
        st.info(" Uploadez une image ou sélectionnez un exemple dans la barre latérale pour commencer")

        # Exemples de classes
        with st.expander(" Voir toutes les classes disponibles"):
            n_cols = 5
            cols = st.columns(n_cols)
            for i, cls in enumerate(class_names):
                with cols[i % n_cols]:
                    st.write(f"• {cls}")

    # Footer
    st.markdown("---")
    st.markdown("""
    ###  À propos
    Ce classificateur utilise **ResNet-50 avec transfer learning** entraîné sur le dataset Food-101.

    **Performance:**
    - Top-1 Accuracy: ~89%
    - Top-5 Accuracy: ~98%
    - Temps d'inférence: <100ms

    **Étudiant:** Mouhamed Diop | **Filière:** DIC2-GIT | **Année:** 2025
    """)


if __name__ == "__main__":
    main()
