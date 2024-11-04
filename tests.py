import unittest
import numpy as np
from data import load_data
from layers import Layer
from network import NeuralNetwork

# Test_load_data: Vérifie que la fonction load_data retourne les données XOR correctes.
# Test_layer_forward: Vérifie que la méthode forward de la classe Layer produit une sortie de la bonne forme et dans la plage attendue pour une fonction d'activation sigmoïde. 
# Test_compute_loss: Vérifie que la méthode compute_loss de la classe NeuralNetwork calcule correctement l'erreur quadratique moyenne.)

class TestMLP(unittest.TestCase):

    def test_load_data(self):
        X, y = load_data()
        expected_X = np.array([[0, 0, 1, 1],
                               [0, 1, 0, 1]])
        expected_y = np.array([[0, 1, 1, 0]])
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)

    def test_layer_forward(self):
        input_data = np.array([[0.5], [0.5]])
        layer = Layer(input_size=2, output_size=1, activation_func='sigmoid')
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue((output >= 0).all() and (output <= 1).all())  # Sigmoid output should be between 0 and 1

    def test_compute_loss(self):
        nn = NeuralNetwork(layer_sizes=[2, 2, 1], activation_function='sigmoid')
        y_true = np.array([[0, 1, 1, 0]])
        y_pred = np.array([[0.1, 0.9, 0.8, 0.2]])
        loss = nn.compute_loss(y_true, y_pred)
        expected_loss = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(loss, expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()