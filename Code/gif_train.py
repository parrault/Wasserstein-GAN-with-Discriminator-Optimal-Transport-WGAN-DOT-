from PIL import Image
import os

def normalize_to_grayscale(image):
    """
    Convertit une image en niveaux de gris et applique une normalisation type MNIST.
    """
    # Convertir l'image en niveaux de gris
    grayscale_image = image.convert("L")
    # Redimensionner à une taille standard MNIST (28x28, facultatif)
    grayscale_image = grayscale_image.resize((28, 28))
    return grayscale_image

def create_gif_from_folder_with_normalization(folder_path, output_gif, duration=500):
    """
    Crée un GIF à partir des images dans un dossier après normalisation en niveaux de gris.

    :param folder_path: Chemin du dossier contenant les images
    :param output_gif: Nom du fichier GIF de sortie
    :param duration: Durée (en ms) entre chaque image dans le GIF
    """
    # Récupérer la liste des fichiers dans le dossier
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    )
    
    if not files:
        print("Aucune image trouvée dans le dossier.")
        return

    # Charger et normaliser les images
    images = [normalize_to_grayscale(Image.open(os.path.join(folder_path, file))) for file in files]

    # Créer le GIF
    images[0].save(
        output_gif, 
        save_all=True, 
        append_images=images[1:], 
        duration=duration, 
        loop=0
    )

    print(f"GIF créé avec succès : {output_gif}")

# Exemple d'utilisation
create_gif_from_folder_with_normalization("samples_latent", "outputtrain.gif", duration=300)