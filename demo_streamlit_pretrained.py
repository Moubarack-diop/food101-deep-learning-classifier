"""
Application Streamlit avec modèle pré-entraîné
POUR DÉMO RAPIDE SANS CHECKPOINT
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title="Food-101 Classifier - Démo",
    page_icon="🍕",
    layout="wide"
)

# Classes Food-101
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

@st.cache_resource
def load_model():
    """Charge ResNet50 pré-entraîné"""
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 101)
    model.eval()
    return model

@st.cache_data
def get_transform():
    """Transformations"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(image, model, transform, top_k=5):
    """Prédiction"""
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            'class': FOOD_CLASSES[idx].replace('_', ' ').title(),
            'probability': prob.item() * 100
        })

    return predictions

# Interface
st.title("🍕 Food-101 Classifier - Version Démo")
st.markdown("### Classification automatique de 101 types d'aliments")
st.info("⚠️ **Version démo** : Utilise un modèle pré-entraîné (non fine-tuné sur Food-101)")

# Sidebar
st.sidebar.header("⚙️ Configuration")

with st.spinner("Chargement du modèle..."):
    model = load_model()
    transform = get_transform()

st.sidebar.success("✅ Modèle chargé")

top_k = st.sidebar.slider("Nombre de prédictions", 1, 10, 5)

# Upload
st.sidebar.header("📤 Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choisir une image",
    type=['jpg', 'jpeg', 'png']
)

# Exemples
st.sidebar.header("🖼️ Ou essayer un exemple")
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

# Charger image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.subheader("📷 Image uploadée")
        st.image(image, use_column_width=True)

elif selected_example != "Aucun":
    try:
        response = requests.get(example_images[selected_example])
        image = Image.open(BytesIO(response.content)).convert('RGB')
        with col1:
            st.subheader("📷 Image d'exemple")
            st.image(image, use_column_width=True)
    except:
        st.error("Erreur lors du chargement de l'exemple")

# Prédiction
if image is not None:
    with col2:
        st.subheader("🎯 Prédictions")

        with st.spinner("Classification en cours..."):
            predictions = predict(image, model, transform, top_k)

        st.success("✅ Classification terminée")

        # Top prédiction
        top_pred = predictions[0]
        st.markdown(f"### 🏆 Prédiction principale")
        st.markdown(f"**{top_pred['class']}**")
        st.progress(top_pred['probability'] / 100)
        st.markdown(f"**Confiance:** {top_pred['probability']:.2f}%")

        # Top-k
        st.markdown(f"### 📊 Top-{top_k} Prédictions")
        for i, pred in enumerate(predictions, 1):
            with st.expander(f"{i}. {pred['class']} - {pred['probability']:.2f}%"):
                st.progress(pred['probability'] / 100)

else:
    st.info("📤 Uploadez une image ou sélectionnez un exemple dans la barre latérale")

# Footer
st.markdown("---")
st.markdown("""
### 📝 À propos de ce projet

**Architecture:** ResNet-50 avec Transfer Learning
**Dataset:** Food-101 (101 classes, 101,000 images)
**Objectif:** Dépasser 85% de précision Top-1

**Résultats attendus avec fine-tuning:**
- Version 2: 66.43% (baseline optimisée)
- Version 2.1: 75-78% (configuration améliorée)
- Version 3: 85-90% (EfficientNet-B4)

**Étudiant:** Mouhamed Diop | **Filière:** DIC2-GIT | **Année:** 2025
""")
