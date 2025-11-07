"""
Script de démo rapide pour la présentation
Utilise un modèle pré-entraîné sans réentraînement
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Charger ResNet50 pré-entraîné
print("🔄 Chargement du modèle pré-entraîné...")
model = models.resnet50(pretrained=True)

# Modifier la dernière couche pour Food-101 (101 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 101)
model.eval()

# Classes Food-101 (extrait)
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

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path_or_url, top_k=5):
    """Prédire la classe d'une image"""

    # Charger l'image
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    # Prétraiter
    img_tensor = transform(image).unsqueeze(0)

    # Prédire
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    # Afficher les résultats
    print(f"\n🍽️  Top-{top_k} Prédictions:")
    print("=" * 50)
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        class_name = FOOD_CLASSES[idx].replace('_', ' ').title()
        print(f"{i}. {class_name:<30} {prob.item()*100:>6.2f}%")

    return top_indices[0][0].item()

if __name__ == "__main__":
    print("🍕 DÉMO CLASSIFICATION FOOD-101")
    print("=" * 50)

    # Exemples d'images
    examples = {
        "Pizza": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400",
        "Sushi": "https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400",
        "Burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400",
    }

    for name, url in examples.items():
        print(f"\n\n📸 Test avec: {name}")
        try:
            predict_image(url)
        except Exception as e:
            print(f"❌ Erreur: {e}")

    print("\n\n✅ Démo terminée!")
    print("💡 Pour la présentation: montrer ce script + architecture + résultats attendus")
