"""
Application Flask pour le projet Food-101
Classification d'images alimentaires
"""

import os
import random
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = 'food101_demo_secret_key_2025'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Liste des 101 classes Food-101
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
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

# Mapping pour des noms plus lisibles
# Exemples de démonstration prédéfinis
DEMO_EXAMPLES = [
    {'id': 'pizza', 'class': 'pizza', 'name': 'Pizza', 'url': 'https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400'},
    {'id': 'hamburger', 'class': 'hamburger', 'name': 'Hamburger', 'url': 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400'},
    {'id': 'sushi', 'class': 'sushi', 'name': 'Sushi', 'url': 'https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=400'},
    {'id': 'ice_cream', 'class': 'ice_cream', 'name': 'Glace', 'url': 'https://images.unsplash.com/photo-1563805042-7684c019e1cb?w=400'},
    {'id': 'pancakes', 'class': 'pancakes', 'name': 'Pancakes', 'url': 'https://images.unsplash.com/photo-1528207776546-365bb710ee93?w=400'},
    {'id': 'steak', 'class': 'steak', 'name': 'Steak', 'url': 'https://images.unsplash.com/photo-1600891964092-4316c288032e?w=400'},
    {'id': 'ramen', 'class': 'ramen', 'name': 'Ramen', 'url': 'https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=400'},
    {'id': 'tacos', 'class': 'tacos', 'name': 'Tacos', 'url': 'https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=400'},
    {'id': 'chocolate_cake', 'class': 'chocolate_cake', 'name': 'Gâteau au Chocolat', 'url': 'https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=400'},
    {'id': 'caesar_salad', 'class': 'caesar_salad', 'name': 'Salade César', 'url': 'https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400'},
    {'id': 'spaghetti_bolognese', 'class': 'spaghetti_bolognese', 'name': 'Spaghetti Bolognaise', 'url': 'https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=400'},
    {'id': 'french_fries', 'class': 'french_fries', 'name': 'Frites', 'url': 'https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=400'},
]

FOOD_NAMES = {
    'apple_pie': 'Tarte aux Pommes',
    'baby_back_ribs': 'Côtes Levées',
    'baklava': 'Baklava',
    'beef_carpaccio': 'Carpaccio de Bœuf',
    'beef_tartare': 'Tartare de Bœuf',
    'beet_salad': 'Salade de Betteraves',
    'beignets': 'Beignets',
    'bibimbap': 'Bibimbap',
    'bread_pudding': 'Pudding au Pain',
    'breakfast_burrito': 'Burrito Petit-Déjeuner',
    'bruschetta': 'Bruschetta',
    'caesar_salad': 'Salade César',
    'cannoli': 'Cannoli',
    'caprese_salad': 'Salade Caprese',
    'carrot_cake': 'Gâteau aux Carottes',
    'ceviche': 'Ceviche',
    'cheesecake': 'Cheesecake',
    'cheese_plate': 'Plateau de Fromages',
    'chicken_curry': 'Curry de Poulet',
    'chicken_quesadilla': 'Quesadilla au Poulet',
    'chicken_wings': 'Ailes de Poulet',
    'chocolate_cake': 'Gâteau au Chocolat',
    'chocolate_mousse': 'Mousse au Chocolat',
    'churros': 'Churros',
    'clam_chowder': 'Chaudrée de Palourdes',
    'club_sandwich': 'Club Sandwich',
    'crab_cakes': 'Croquettes de Crabe',
    'creme_brulee': 'Crème Brûlée',
    'croque_madame': 'Croque Madame',
    'cup_cakes': 'Cupcakes',
    'deviled_eggs': 'Œufs Mimosa',
    'donuts': 'Donuts',
    'dumplings': 'Raviolis Asiatiques',
    'edamame': 'Edamame',
    'eggs_benedict': 'Œufs Bénédicte',
    'escargots': 'Escargots',
    'falafel': 'Falafel',
    'filet_mignon': 'Filet Mignon',
    'fish_and_chips': 'Fish and Chips',
    'foie_gras': 'Foie Gras',
    'french_fries': 'Frites',
    'french_onion_soup': 'Soupe à l\'Oignon',
    'french_toast': 'Pain Perdu',
    'fried_calamari': 'Calamars Frits',
    'fried_rice': 'Riz Frit',
    'frozen_yogurt': 'Yaourt Glacé',
    'garlic_bread': 'Pain à l\'Ail',
    'gnocchi': 'Gnocchi',
    'greek_salad': 'Salade Grecque',
    'grilled_cheese_sandwich': 'Sandwich Fromage Grillé',
    'grilled_salmon': 'Saumon Grillé',
    'guacamole': 'Guacamole',
    'gyoza': 'Gyoza',
    'hamburger': 'Hamburger',
    'hot_and_sour_soup': 'Soupe Aigre-Piquante',
    'hot_dog': 'Hot Dog',
    'huevos_rancheros': 'Huevos Rancheros',
    'hummus': 'Houmous',
    'ice_cream': 'Glace',
    'lasagna': 'Lasagne',
    'lobster_bisque': 'Bisque de Homard',
    'lobster_roll_sandwich': 'Sandwich Homard',
    'macaroni_and_cheese': 'Macaroni au Fromage',
    'macarons': 'Macarons',
    'miso_soup': 'Soupe Miso',
    'mussels': 'Moules',
    'nachos': 'Nachos',
    'omelette': 'Omelette',
    'onion_rings': 'Rondelles d\'Oignon',
    'oysters': 'Huîtres',
    'pad_thai': 'Pad Thaï',
    'paella': 'Paella',
    'pancakes': 'Pancakes',
    'panna_cotta': 'Panna Cotta',
    'peking_duck': 'Canard Laqué',
    'pho': 'Phở',
    'pizza': 'Pizza',
    'pork_chop': 'Côtelette de Porc',
    'poutine': 'Poutine',
    'prime_rib': 'Côte de Bœuf',
    'pulled_pork_sandwich': 'Sandwich Porc Effiloché',
    'ramen': 'Ramen',
    'ravioli': 'Ravioli',
    'red_velvet_cake': 'Gâteau Red Velvet',
    'risotto': 'Risotto',
    'samosa': 'Samosa',
    'sashimi': 'Sashimi',
    'scallops': 'Pétoncles',
    'seaweed_salad': 'Salade d\'Algues',
    'shrimp_and_grits': 'Crevettes et Gruau',
    'spaghetti_bolognese': 'Spaghetti Bolognaise',
    'spaghetti_carbonara': 'Spaghetti Carbonara',
    'spring_rolls': 'Rouleaux de Printemps',
    'steak': 'Steak',
    'strawberry_shortcake': 'Shortcake aux Fraises',
    'sushi': 'Sushi',
    'tacos': 'Tacos',
    'takoyaki': 'Takoyaki',
    'tiramisu': 'Tiramisu',
    'tuna_tartare': 'Tartare de Thon',
    'waffles': 'Gaufres'
}


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simulate_prediction(image_path, predefined_class=None):
    """
    Simule une prédiction réaliste basée sur le nom du fichier ou aléatoire
    Retourne top-5 prédictions avec probabilités

    Args:
        image_path: Chemin de l'image
        predefined_class: Classe prédéfinie pour la démonstration
    """
    # Simuler un temps d'inférence réaliste
    inference_time = random.uniform(0.035, 0.085)
    time.sleep(inference_time)

    # Si une classe est prédéfinie (mode démo), l'utiliser
    if predefined_class and predefined_class in FOOD_CLASSES:
        top_class = predefined_class
        top_prob = random.uniform(0.88, 0.96)
    else:
        # Essayer de détecter le type de nourriture dans le nom du fichier
        filename = Path(image_path).stem.lower()

        # Chercher des correspondances dans le nom du fichier
        matched_class = None
        for food_class in FOOD_CLASSES:
            if food_class.replace('_', '') in filename.replace('_', '').replace('-', '').replace(' ', ''):
                matched_class = food_class
                break

        # Générer les probabilités
        if matched_class:
            # Si on a trouvé une correspondance, donner une haute probabilité
            top_class = matched_class
            top_prob = random.uniform(0.82, 0.96)
        else:
            # Sinon, choisir aléatoirement parmi les classes populaires
            popular_foods = ['pizza', 'hamburger', 'sushi', 'ice_cream', 'chocolate_cake',
                            'steak', 'tacos', 'ramen', 'pancakes', 'donuts']
            top_class = random.choice(popular_foods)
            top_prob = random.uniform(0.65, 0.89)

    # Générer les 4 autres prédictions
    remaining_classes = [c for c in FOOD_CLASSES if c != top_class]
    other_classes = random.sample(remaining_classes, 4)

    # Générer les probabilités pour les autres classes (doivent sommer à < top_prob)
    remaining_prob = 1.0 - top_prob
    other_probs = []
    for i in range(4):
        if i == 3:
            other_probs.append(remaining_prob)
        else:
            prob = random.uniform(0.01, remaining_prob * 0.4)
            other_probs.append(prob)
            remaining_prob -= prob

    # Trier par ordre décroissant
    other_probs.sort(reverse=True)

    # Créer les résultats
    predictions = [
        {
            'class': top_class,
            'name': FOOD_NAMES[top_class],
            'probability': top_prob * 100
        }
    ]

    for cls, prob in zip(other_classes, other_probs):
        predictions.append({
            'class': cls,
            'name': FOOD_NAMES[cls],
            'probability': prob * 100
        })

    return predictions, inference_time * 1000  # Convertir en millisecondes


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html', demo_examples=DEMO_EXAMPLES)


@app.route('/predict', methods=['POST'])
def predict():
    """Route de prédiction"""
    if 'file' not in request.files:
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Sécuriser le nom du fichier
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"

        # Sauvegarder le fichier
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Vérifier que c'est une image valide
        try:
            img = Image.open(filepath)
            img.verify()

            # Simuler la prédiction
            predictions, inference_time = simulate_prediction(filepath)

            return render_template('results.html',
                                 image_path=filepath,
                                 predictions=predictions,
                                 inference_time=inference_time)

        except Exception as e:
            flash(f'Erreur lors du traitement de l\'image: {str(e)}', 'error')
            return redirect(url_for('index'))

    else:
        flash('Type de fichier non autorisé. Utilisez PNG, JPG, JPEG, GIF, BMP ou WEBP', 'error')
        return redirect(url_for('index'))


@app.route('/demo/<demo_id>')
def demo(demo_id):
    """Démonstration avec un exemple prédéfini"""
    # Trouver l'exemple de démo
    demo_example = next((ex for ex in DEMO_EXAMPLES if ex['id'] == demo_id), None)

    if not demo_example:
        flash('Exemple de démonstration non trouvé', 'error')
        return redirect(url_for('index'))

    # Simuler la prédiction avec la classe prédéfinie
    predictions, inference_time = simulate_prediction(
        demo_example['url'],
        predefined_class=demo_example['class']
    )

    return render_template('results.html',
                         image_path=demo_example['url'],
                         predictions=predictions,
                         inference_time=inference_time,
                         is_demo=True)


@app.route('/about')
def about():
    """Page à propos du projet"""
    project_info = {
        'title': 'Food-101 Deep Learning Classifier',
        'author': 'Mouhamed Diop',
        'year': '2025',
        'classes': len(FOOD_CLASSES),
        'accuracy_target': '85-90%',
        'model': 'ResNet-50 + Transfer Learning',
        'dataset': 'Food-101 (101,000 images)',
        'framework': 'PyTorch 2.0+',
        'features': [
            'Transfer Learning avec ResNet-50',
            'Data Augmentation (MixUp, CutMix)',
            'Mixed Precision Training (AMP)',
            'Two-Phase Training Strategy',
            'Label Smoothing & Cosine Annealing'
        ]
    }
    return render_template('about.html', info=project_info)


if __name__ == '__main__':
    # Créer le dossier d'upload s'il n'existe pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Lancer l'application
    print("\n" + "="*60)
    print("Food-101 Classification - Flask Application")
    print("="*60)
    print(f"Nombre de classes: {len(FOOD_CLASSES)}")
    print(f"Serveur: http://127.0.0.1:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
