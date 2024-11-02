# Implémentation de la Rétropropagation dans un MLP

## Description

Ce projet implémente un réseau de type MLP. Il est conçu pour résoudre un problème de classification binaire, non linéaire, simples (fonction XOR).

## Structure du Projet

- `main.py`: Point d'entrée du programme.
- `network.py`: Classe du réseau de neurones.
- `layers.py`: Classe pour les couches du réseau.
- `activations.py`: Fonctions d'activation et leurs dérivées.
- `data.py`: Chargement des données.
- `config.py`: Configuration et hyperparamètres.
- `README.md`: Instructions et informations sur le projet.
- `requirements.txt`: Dépendances requises.
- `combine.py`: Combinaison des résultats des différentes configurations.
## Prérequis

- Python 3.12
- Numpy

## Installation

1. Télécharger les fichiers.
2. Installer les dépendances avec la commande :

```bash
pip install -r requirements.txt
```

## Configuration

- `config.py` : Configuration et hyperparamètres.

## Exécution

- Exécuter avec autant de configurations que voulu le programme avec la commande :
```bash
python main.py
```
- Combinez les résultats des différentes configurations avec la commande :
```bash
python combine.py
```

## Résultats

- Plusieurs configurations ont été testées :
    - Résultats enregistrés dans le dossier `models`
    - Les dossiers sont nommés selon la configuration: `<fonction d'activation>_hl=<liste des nombres de neurones par couche cachée>` et contiennent les fichiers `classification_plane.png` et `loss.png`
    - Une combinaison des résultats des différentes configurations est présente dans le dossier `models/combined` et est nommée `combined_<fonction d'activation>_classifications.png`.
    - Fonctions d'activation testées : sigmoid, relu
    - Taux d'apprentissage : dépend de la configuration.

- Initialisation Xavier pour les poids des couches.

- Relu :
    - Le taux d'apprentissage est très dur à régler pour la fonction relu. On peut réduire le taux d'apprentissage pour ne pas avoir une loss constante égale à 0.5, mais la loss diminue très lentement au bout d'un certain temps. Une méthode de descente de gradient adaptative pourrait être plus efficace.

![Plans de classification ReLU](models/combined_relu_classifications.png)

- Sigmoid :
    - Les résultats sont biens meilleurs avec la fonction sigmoid, mais on voit qu'avec la configuration [2, 2, 2, 2, 2, 2] On converge vers un minimum local de la loss qui vaut environ 0.25 et ce peut importe le taux d'apprentissage. C'est un problème de vanishing gradient dont la fonction sigmoid est sensible lorsque le nombre de couches cachées augmente.

![Plans de classification Sigmoid](models/combined_sigmoid_classifications.png)