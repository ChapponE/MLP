# layers.py
import numpy as np
import activations
import config

class Layer:
    def __init__(self, input_size, output_size, activation_func='sigmoid'):
        """
        Initialise une couche du réseau.
        """
        if activation_func == 'sigmoid':
            # Initialisation Xavier pour sigmoid
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        elif activation_func == 'relu':
            # Initialisation He pour ReLU
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        else:
            # Initialisation par défaut
            self.weights = np.random.randn(output_size, input_size) * 0.1
            
        self.biases = np.zeros((output_size, 1))
        self.delta = None
        self.z = None
        self.a = None
        self.activation_func = activation_func
        self.activation = getattr(activations, self.activation_func)
        self.activation_derivative = getattr(activations, self.activation_func + '_derivative')

    def forward(self, input_data):
        """
        Calcul de la propagation avant pour cette couche.

        Parameters:
        - input_data: Données d'entrée pour cette couche.

        Returns:
        - output: Sortie après activation.
        """
        self.z = np.dot(self.weights, input_data) + self.biases
        self.a = self.activation(self.z)
        return self.a
