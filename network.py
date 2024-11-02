# network.py
import numpy as np
import matplotlib.pyplot as plt
import os
from layers import Layer
from config import CONFIG

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function='sigmoid'):
        """
        Initialise le réseau de neurones.
        """
        self.layers = []
        self.activation_function = activation_function
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(input_size=layer_sizes[i],
                                     output_size=layer_sizes[i+1],
                                     activation_func=self.activation_function))

    def forward(self, X):
        """
        Propagation avant à travers le réseau.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def compute_loss(self, y_true, y_pred):
        """
        Calcule l'erreur des moindres carrés.
        """
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self, X, y_true):
        """
        Rétropropagation du gradient.
        """
        # Propagation avant
        output = self.forward(X)

        # Calcul de l'erreur
        error = y_true - output

        # Calcul du gradient pour la dernière couche
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                layer.delta = error * layer.activation_derivative(layer.z)
            else:
                next_layer = self.layers[i + 1]
                layer.delta = np.dot(next_layer.weights.T, next_layer.delta) * layer.activation_derivative(layer.z)

            # Normalisation du gradient si activée
            if CONFIG['normalize_gradient']:
                # Calcul de la norme du gradient
                gradient_norm = np.linalg.norm(layer.delta)
                if gradient_norm > 1:
                    layer.delta = layer.delta / gradient_norm

    def update_weights(self, X, learning_rate):
        """
        Met à jour les poids et biais du réseau.
        """
        input = X
        for layer in self.layers:
            layer.weights += learning_rate * np.dot(layer.delta, input.T)
            layer.biases += learning_rate * np.sum(layer.delta, axis=1, keepdims=True)
            input = layer.a

    def train(self, X, y, learning_rate=0.1, threshold=0.01, max_epochs=10000, model_dir=None):
        """
        Entraîne le réseau de neurones, calcule la perte, et l'afffiche.
        """
        self.loss_history = []
        self.training_data = (X, y)
        
        # Variables pour l'early stopping
        best_loss = float('inf')
        patience = 1000  # Nombre d'époques sans amélioration avant arrêt
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):
            self.backward(X, y)
            self.update_weights(X, learning_rate)
            
            loss = self.compute_loss(y, self.forward(X))
            self.loss_history.append(loss)
            
            if epoch % 1000 == 0:
                print(f"Époque {epoch}, Perte: {loss}")

            # Vérifier s'il y a une amélioration
            if loss < best_loss - threshold:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            # Early stopping si pas d'amélioration pendant 'patience' époques
            if epochs_without_improvement >= patience:
                print(f"Early stopping à l'époque {epoch}: pas d'amélioration depuis {patience} époques")
                break
                
            # Arrêt si la perte est suffisamment faible
            if loss < threshold:
                print(f"Convergence atteinte à l'époque {epoch}")
                break

        # Tracer et sauvegarder l'évolution de la perte
        plt.plot(self.loss_history)
        plt.title("Évolution de la perte au cours des époques")
        plt.xlabel("Époque")
        plt.ylabel("Perte")
        if model_dir:
            loss_plot_path = os.path.join(model_dir, 'loss.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Graphique de perte sauvegardé dans {loss_plot_path}")
            plt.show()
        else:
            plt.show()

    def predict(self, X):
        """
        Prédit les sorties.
        """
        output = self.forward(X)
        return output > 0.5  # Seuil pour la classification binaire
    
    def plot_classification_plane(self, x_range, y_range, model_dir=None):
        """
        Affiche les frontières de décisions.
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Shape (2, N)

        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=[-0.1, 0.5, 1.1], colors=['lightblue', 'lightcoral'], alpha=0.6)

        # Plot des données d'entraînement
        X_train, y_train = self.training_data
        plt.scatter(X_train[0, :], X_train[1, :], c=y_train.ravel(), edgecolors='k', cmap=plt.cm.Paired)

        plt.title('Plan de classification')
        plt.xlabel('x₁')
        plt.ylabel('x₂')

        if model_dir:
            classification_plot_path = os.path.join(model_dir, 'classification_plane.png')
            plt.savefig(classification_plot_path)
            plt.close()
            print(f"Graphique du plan de classification sauvegardé dans {classification_plot_path}")
            plt.show()
        else:
            plt.show()
