# config.py
CONFIG = {
    'layer_sizes': [2, 2,2,2,2,2,2, 1],  # entrées, nb de neurones des couches cachées, une sortie
    'learning_rate': 0.1,
    'threshold': 0.000001,
    'max_epochs': 100000,
    'activation_function': 'sigmoid'  # Choix de 'sigmoid' ou 'relu'
}