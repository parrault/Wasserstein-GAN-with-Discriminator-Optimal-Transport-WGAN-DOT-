import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def open_and_normalize_image(folder, number):
    # Construire le chemin vers l'image
    image_path = os.path.join(folder, f"{number}.png")
    
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {number}.jpg n'existe pas dans le dossier {folder}.")
        return
    
    # Ouvrir et normaliser l'image
    try:
        img = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
        img_array = np.array(img, dtype=np.float32)  # Convertir en tableau numpy
        normalized_img = img_array / 255.0  # Normaliser entre 0 et 1
        
        print(f"L'image {number}.jpg a été normalisée entre 0 et 1.")
        
        # Afficher l'image normalisée
        plt.imshow(normalized_img, cmap="gray")
        plt.title(f"Image {number}.jpg (normalisée)")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Erreur lors de l'ouverture ou de la normalisation de l'image : {e}")

if __name__ == "__main__":
    # Dossier contenant les images
    folder_path = "./samples"  # Modifier si nécessaire

    # Demander le numéro de l'image
    number = input("Entrez le numéro de l'image (sans extension) : ").strip()
    
    # Vérifier que l'entrée est un nombre
    if not number.isdigit():
        print("Erreur : Vous devez entrer un numéro valide.")
    else:
        open_and_normalize_image(folder_path, int(number))

