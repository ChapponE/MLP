# main.py
import os
from network import NeuralNetwork
from data import load_data
from config import CONFIG

def main():
    # Chargement des données
    X, y = load_data()

    # Initialisation du réseau de neurones
    nn = NeuralNetwork(CONFIG['layer_sizes'],
                       activation_function=CONFIG['activation_function'])

    # Détermination du dossier du modèle
    activation_func = CONFIG['activation_function']
    hidden_layers_sizes = CONFIG['layer_sizes'][1:-1]

    # Ajout du suffixe 'normalized' si la normalisation est activée
    normalized = '_normalized' if CONFIG['normalize_gradient'] else ''

    model_dir = os.path.join('models', f"{activation_func}{normalized}_hl={hidden_layers_sizes}")
    os.makedirs(model_dir, exist_ok=True)

    # Entraînement du réseau
    nn.train(X, y,
             learning_rate=CONFIG['learning_rate'],
             threshold=CONFIG['threshold'],
             max_epochs=CONFIG['max_epochs'],
             model_dir=model_dir)

    # Test du réseau
    predictions = nn.predict(X)
    print("Prédictions:", predictions)
    print("Vraies valeurs:", y)

    # Affichage du plan de classification
    nn.plot_classification_plane(x_range=(-2, 2), y_range=(-2, 2), model_dir=model_dir)

if __name__ == "__main__":
    main()
