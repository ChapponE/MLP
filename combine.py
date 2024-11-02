import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG

# Paramètre pour choisir le type d'activation
activation_type = CONFIG['ACTIVATION_FUNCTION']

# Chemin vers le dossier models
models_dir = 'models'

# Récupérer tous les dossiers commençant par le type d'activation choisi
model_dirs = glob.glob(os.path.join(models_dir, f'{activation_type}*'))

# Trier les dossiers par nom pour avoir un ordre cohérent
model_dirs.sort()

# Nombre d'images trouvées
n_images = len(model_dirs)

# Calculer le nombre de lignes et colonnes pour la grille
n_cols = 3  # Vous pouvez ajuster ce nombre
n_rows = (n_images + n_cols - 1) // n_cols

# Créer une figure avec des sous-plots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
fig.suptitle(f'Comparaison des plans de classification pour différentes architectures {activation_type.upper()}', 
             fontsize=16)

# Aplatir axes si nécessaire pour faciliter l'itération
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# Pour chaque dossier
for idx, model_dir in enumerate(model_dirs):
    # Calculer la position dans la grille
    row = idx // n_cols
    col = idx % n_cols
    
    # Charger l'image
    img_path = os.path.join(model_dir, 'classification_plane.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[row, col].imshow(img)
        
        # Extraire le nom de l'architecture à partir du nom du dossier
        arch_name = os.path.basename(model_dir).replace(f'{activation_type}_hl=', '')
        axes[row, col].set_title(f'Architecture: {arch_name}')
        axes[row, col].axis('off')

# Masquer les sous-plots vides
for idx in range(len(model_dirs), n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].axis('off')

# Ajuster l'espacement entre les sous-plots
plt.tight_layout()

# Sauvegarder la figure combinée
output_filename = f'combined_{activation_type}_classifications.png'
plt.savefig(os.path.join(models_dir, output_filename))
plt.close()

print(f"Image combinée sauvegardée dans 'models/{output_filename}'")