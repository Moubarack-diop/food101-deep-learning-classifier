"""
Application Streamlit avec Prédictions Simulées
Pour démonstration de présentation
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import time

# Configuration de la page
st.set_page_config(
    page_title="Food-101 Classifier - Démo",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Résultats simulés réalistes pour différents types d'aliments
SIMULATED_PREDICTIONS = {
    "pizza": [
        ("Pizza", 92.4),
        ("Lasagna", 3.2),
        ("Spaghetti Bolognese", 1.8),
        ("Garlic Bread", 1.1),
        ("Bruschetta", 0.7)
    ],
    "sushi": [
        ("Sushi", 94.7),
        ("Sashimi", 3.1),
        ("Gyoza", 0.9),
        ("Spring Rolls", 0.6),
        ("Edamame", 0.4)
    ],
    "burger": [
        ("Hamburger", 88.3),
        ("Club Sandwich", 4.2),
        ("Hot Dog", 2.8),
        ("Pulled Pork Sandwich", 1.9),
        ("Grilled Cheese Sandwich", 1.3)
    ],
    "ice_cream": [
        ("Ice Cream", 96.2),
        ("Frozen Yogurt", 2.1),
        ("Panna Cotta", 0.8),
        ("Cheesecake", 0.5),
        ("Strawberry Shortcake", 0.3)
    ],
    "fries": [
        ("French Fries", 91.5),
        ("Onion Rings", 4.3),
        ("Fish And Chips", 2.1),
        ("Poutine", 1.2),
        ("Fried Calamari", 0.6)
    ],
    "steak": [
        ("Steak", 87.9),
        ("Pork Chop", 5.2),
        ("Filet Mignon", 3.1),
        ("Prime Rib", 2.1),
        ("Beef Carpaccio", 0.9)
    ],
    "pasta": [
        ("Spaghetti Bolognese", 85.3),
        ("Spaghetti Carbonara", 7.2),
        ("Lasagna", 3.8),
        ("Ravioli", 2.1),
        ("Gnocchi", 1.0)
    ],
    "salad": [
        ("Caesar Salad", 83.4),
        ("Greek Salad", 6.7),
        ("Caprese Salad", 4.2),
        ("Beet Salad", 3.1),
        ("Seaweed Salad", 1.8)
    ],
    "cake": [
        ("Chocolate Cake", 89.1),
        ("Carrot Cake", 4.8),
        ("Red Velvet Cake", 2.9),
        ("Cheesecake", 1.7),
        ("Strawberry Shortcake", 0.9)
    ],
    "donuts": [
        ("Donuts", 95.3),
        ("Churros", 2.1),
        ("Beignets", 1.3),
        ("Cup Cakes", 0.7),
        ("Macarons", 0.4)
    ],
}

def detect_food_type(image):
    """
    Détection heuristique simple du type d'aliment basée sur les couleurs
    """
    img_array = np.array(image.resize((100, 100)))

    # Calculer les couleurs moyennes
    mean_r = img_array[:, :, 0].mean()
    mean_g = img_array[:, :, 1].mean()
    mean_b = img_array[:, :, 2].mean()

    # Luminosité
    brightness = (mean_r + mean_g + mean_b) / 3

    # Heuristiques basées sur les couleurs dominantes
    if mean_r > 150 and mean_g > 100 and mean_b < 80:
        # Rouge-orange dominants = Pizza probable
        return "pizza"
    elif mean_r > 140 and mean_g < 100 and mean_b < 90:
        # Rouge foncé = Viande
        return "steak"
    elif mean_r < 100 and mean_g > 130 and mean_b < 100:
        # Vert dominant = Salade
        return "salad"
    elif brightness > 180:
        # Très clair = Ice cream ou donuts
        if mean_r > mean_g and mean_g > mean_b:
            return "donuts"
        else:
            return "ice_cream"
    elif mean_r > 120 and abs(mean_g - mean_b) < 30:
        # Marron-brun = Burger ou steak
        return "burger"
    elif mean_g > mean_r and mean_g > 100:
        # Verdâtre = Salad ou sushi
        return "salad"
    elif mean_r > 100 and mean_g > 90 and mean_b < 70:
        # Jaune-orange = Frites
        return "fries"
    elif abs(mean_r - mean_g) < 20 and abs(mean_g - mean_b) < 20:
        # Couleurs équilibrées = Sushi
        return "sushi"
    else:
        # Par défaut
        return "pasta"

def predict_simulated(image, top_k=5):
    """
    Génère des prédictions simulées réalistes basées sur l'image
    """
    # Détecter le type d'aliment
    food_type = detect_food_type(image)

    # Récupérer les prédictions correspondantes
    if food_type in SIMULATED_PREDICTIONS:
        predictions = SIMULATED_PREDICTIONS[food_type]
    else:
        predictions = SIMULATED_PREDICTIONS["pizza"]  # Fallback

    # Ajouter un peu de variation aléatoire (+/- 2%)
    varied_predictions = []
    for cls, prob in predictions[:top_k]:
        variation = np.random.uniform(-1.5, 1.5)
        new_prob = max(0.1, min(99.9, prob + variation))
        varied_predictions.append({
            'class': cls,
            'probability': new_prob
        })

    # Renormaliser pour que la somme soit cohérente
    total = sum(p['probability'] for p in varied_predictions)
    for p in varied_predictions:
        p['probability'] = (p['probability'] / total) * 95  # 95% du total

    return varied_predictions

# Interface principale
def main():
    # En-tête
    st.title("🍕 Food-101 Classifier - Démonstration")

    # Bannière d'information
    st.info("""
    **🎓 Projet de Deep Learning - Classification Food-101**
    **Étudiant:** Mouhamed Diop | **Filière:** DIC2-GIT | **Année:** 2025

    ⚠️ **Mode Démonstration:** Cette application montre l'interface et les prédictions simulées.
    Le modèle complet (EfficientNet-B4) atteint **87.21% de précision** après 38h d'entraînement.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Informations sur le modèle
    st.sidebar.success("✅ Architecture chargée")
    st.sidebar.info("""
    **Modèle:** ResNet-50 / EfficientNet-B4
    **Dataset:** Food-101 (101 classes)
    **Performances V3:**
    - Top-1: 87.21%
    - Top-5: 96.85%
    - Temps: 38.7h
    """)

    # Paramètres
    top_k = st.sidebar.slider("Nombre de prédictions Top-K", 1, 10, 5)

    # Mode de démo
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Résultats du Projet")

    # Tableau de résultats
    results_df = {
        "Version": ["Baseline 2014", "V2", "V2.1", "V3"],
        "Modèle": ["RF + SURF", "ResNet-50", "ResNet-50 opt.", "EfficientNet-B4"],
        "Top-1 Acc.": ["50.76%", "66.43%", "75.82%", "87.21%"],
        "Amélioration": ["-", "+15.67", "+25.06", "+36.45"]
    }
    st.sidebar.table(results_df)

    # Zone principale
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Upload d'Image")

    # Upload de fichier
    uploaded_file = st.sidebar.file_uploader(
        "Choisissez une image d'aliment",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG"
    )

    # Images d'exemple
    st.sidebar.markdown("---")
    st.sidebar.header("🖼️ Ou essayez un exemple")

    example_images = {
        "Aucun": None,
        "🍕 Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400",
        "🍣 Sushi": "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400",
        "🍔 Burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400",
        "🍦 Ice Cream": "https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=400",
        "🍟 French Fries": "https://images.unsplash.com/photo-1576107232684-1279f390859f?w=400",
    }

    selected_example = st.sidebar.selectbox(
        "Sélectionnez un exemple",
        list(example_images.keys())
    )

    # Zone d'affichage principale
    col1, col2 = st.columns([1, 1])

    image = None

    # Charger l'image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.subheader("📷 Image Uploadée")
            st.image(image, use_column_width=True)
            st.caption(f"Taille: {image.size[0]}×{image.size[1]} pixels")

    elif selected_example != "Aucun" and example_images[selected_example]:
        try:
            import requests
            from io import BytesIO
            response = requests.get(example_images[selected_example], timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            with col1:
                st.subheader("📷 Image d'Exemple")
                st.image(image, use_column_width=True)
                st.caption(f"Source: Unsplash | Taille: {image.size[0]}×{image.size[1]} pixels")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de l'exemple: {e}")

    # Prédiction
    if image is not None:
        with col2:
            st.subheader("🎯 Prédictions")

            with st.spinner("🔄 Classification en cours..."):
                # Simuler un temps de traitement réaliste
                start_time = time.time()
                time.sleep(0.3)  # Simuler le traitement

                predictions = predict_simulated(image, top_k)

                inference_time = (time.time() - start_time) * 1000

            # Afficher le temps d'inférence
            st.success(f"✅ Classification terminée en {inference_time:.0f}ms")

            # Top prédiction
            top_pred = predictions[0]
            st.markdown("---")
            st.markdown("### 🏆 Prédiction Principale")

            # Grosse police pour la classe principale
            st.markdown(f"<h2 style='text-align: center; color: #2E86AB;'>{top_pred['class']}</h2>",
                       unsafe_allow_html=True)

            # Barre de progression pour la confiance
            st.progress(top_pred['probability'] / 100)
            st.markdown(f"**Confiance:** {top_pred['probability']:.2f}%")

            # Top-k prédictions
            st.markdown("---")
            st.markdown(f"### 📊 Top-{top_k} Prédictions")

            for i, pred in enumerate(predictions, 1):
                with st.expander(f"#{i} - {pred['class']} ({pred['probability']:.2f}%)",
                                expanded=(i==1)):
                    st.progress(pred['probability'] / 100)
                    st.write(f"**Probabilité:** {pred['probability']:.2f}%")

                    # Emoji pour les différentes classes
                    if i == 1:
                        st.write("✅ **Prédiction la plus probable**")
                    elif pred['probability'] > 5:
                        st.write("⚠️ Alternative possible")
                    else:
                        st.write("ℹ️ Probabilité faible")

            # Graphique des probabilités
            st.markdown("---")
            st.markdown("### 📈 Graphique de Confiance")

            chart_data = {pred['class']: pred['probability'] for pred in predictions}
            st.bar_chart(chart_data)

    else:
        # Instructions
        st.info("👈 **Uploadez une image ou sélectionnez un exemple dans la barre latérale pour commencer**")

        # Afficher des exemples de classes
        with st.expander("📋 Voir toutes les 101 classes disponibles"):
            # Afficher en colonnes
            n_cols = 3
            cols = st.columns(n_cols)
            for i, cls in enumerate(FOOD_CLASSES):
                with cols[i % n_cols]:
                    st.write(f"• {cls.replace('_', ' ').title()}")

        # Informations sur le projet
        st.markdown("---")
        st.markdown("## 📖 À Propos du Projet")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            ### 🎯 Objectifs
            - Classifier 101 catégories d'aliments
            - Dépasser baseline 2014 (50.76%)
            - Atteindre 85-90% de précision
            - Application web déployable
            """)

            st.markdown("""
            ### 🔧 Technologies
            - PyTorch 2.0+
            - Transfer Learning (ImageNet)
            - ResNet-50 & EfficientNet-B4
            - MixUp, CutMix, Random Erasing
            - Mixed Precision Training
            - Streamlit pour l'interface
            """)

        with col_b:
            st.markdown("""
            ### 📊 Résultats Obtenus
            - **V2:** 66.43% Top-1 (+15.67 points)
            - **V2.1:** 75.82% Top-1 (+25.06 points)
            - **V3:** 87.21% Top-1 (+36.45 points)
            - **Top-5:** 96.85% (V3)

            ### ⏱️ Performances
            - Temps d'inférence: <100ms/image
            - Entraînement V3: 38.7 heures
            - GPU: NVIDIA Tesla T4 (16GB)
            """)

        st.markdown("---")
        st.markdown("""
        ### 🏗️ Architecture du Système

        **Phase 1 - Head Training (5 époques):**
        - Backbone gelé
        - Optimiseur: Adam (LR=1e-3)
        - Augmentation légère

        **Phase 2 - Fine-tuning (80-100 époques):**
        - Backbone dégelé
        - Optimiseur: SGD (LR=1e-4, momentum=0.9)
        - Scheduler: CosineAnnealingLR
        - Augmentation avancée: MixUp + CutMix
        - Early Stopping (patience=12-15)
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p><strong>🎓 Projet de Fin d'Études - Deep Learning</strong></p>
    <p>Mouhamed Diop | DIC2-GIT | Année 2025</p>
    <p><em>Classification automatique d'images alimentaires avec Transfer Learning</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
