import os
from PIL import Image
import matplotlib.pyplot as plt

# Chemin du dossier contenant les images
folder_path = "samples"
output_file = "grid_output.png"  # Nom du fichier de sortie

# Fonction pour charger et normaliser une image en noir et blanc
def load_and_normalize_image(image_path):
    img = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    img = img.resize((28, 28))  # Redimensionner si nécessaire (MNIST : 28x28)
    return img

# Fonction pour afficher et enregistrer une grille d'images
def create_and_save_image_grid(folder_path, output_file, grid_size=(10, 10)):
    images = []
    
    # Charger toutes les images du dossier
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, file_name)
            images.append(load_and_normalize_image(image_path))
            # Stop si la grille est remplie
            if len(images) >= grid_size[0] * grid_size[1]:
                break
    
    if not images:
        print("Aucune image trouvée dans le dossier spécifié.")
        return
    
    # Taille de chaque image (supposée identique pour toutes)
    img_width, img_height = images[0].size
    
    # Taille totale de la grille
    grid_width = img_width * grid_size[1]
    grid_height = img_height * grid_size[0]
    
    # Créer une grande image vide pour la grille
    grid_image = Image.new('L', (grid_width, grid_height))
    
    # Coller les images dans la grille
    for idx, img in enumerate(images):
        x_offset = (idx % grid_size[1]) * img_width
        y_offset = (idx // grid_size[1]) * img_height
        grid_image.paste(img, (x_offset, y_offset))
    
    # Sauvegarder la grille
    grid_image.save(output_file)
    print(f"Grille sauvegardée dans le fichier : {output_file}")

# Créer une grille de 20x20 (400 images) et enregistrer le résultat
create_and_save_image_grid(folder_path, output_file, grid_size=(20, 20))
