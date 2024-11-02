# data.py
import numpy as np

def load_data():
    """
    Données pour la fonction XOR

    Returns:
    - X: Données d'entrée.
    - y: Valeurs cibles.
    """
    
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    return X, y
