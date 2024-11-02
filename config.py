# config.py
CONFIG = {
    'layer_sizes': [2, 2, 1],  # entrées, nb de neurones des couches cachées, une sortie
    'learning_rate': 0.15,
    'normalize_gradient': True,
    'threshold': 0.00001,
    'max_epochs': 100000,
    'activation_function': 'relu'  # Choix de 'sigmoid' ou 'relu'
}