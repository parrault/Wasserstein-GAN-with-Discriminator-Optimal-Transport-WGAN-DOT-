import torch 
import torchvision
import os
import argparse
import shutil
from PIL import Image
from model import Generator, Discriminator
from utils import load_model_gene, device, load_model_discr


def clean_folder(folder_path):
    """
    Supprime tous les fichiers dans un dossier.
    
    Args:
        folder_path (str): Chemin du dossier à nettoyer.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Supprime le dossier et son contenu
    os.makedirs(folder_path, exist_ok=True)  # Recrée un dossier vide


def latent_space_dot_gif(generator, discriminator, z_y, keff, epsilon=10, n_updates=10, delta=1e-3):
    """
    Applique le transport optimal dans l'espace latent et génère les étapes intermédiaires
    pour un GIF.

    Args:
        generator (torch.nn.Module): Générateur pré-entraîné.
        discriminator (torch.nn.Module): Discriminateur pré-entraîné.
        z (torch.Tensor): Échantillon latent initial.
        keff (float): Constante Lipschitz approximée.
        epsilon (float): Taux d'apprentissage pour la descente de gradient.
        n_updates (int): Nombre d'itérations de mise à jour.
        delta (float): Terme stabilisateur pour éviter les instabilités numériques.

    Returns:
        torch.Tensor: Échantillon latent après optimisation.
        list[torch.Tensor]: Liste des images générées pour chaque étape.
    """
    z = z_y.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=epsilon)
    images = []

    for _ in range(n_updates):
        optimizer.zero_grad()
        
        # Calcul de l'objectif
        term1 = torch.norm(z - z_y + delta, p=2) 
        term2 = -1 / keff * discriminator(generator(z)).mean()
        loss = term1 + term2
        

        # Backpropagation et optimisation
        loss.backward()
        optimizer.step()
        z.data.clamp_(-3, 3)

        # Sauvegarde de l'image générée après chaque mise à jour
        with torch.no_grad():
            generated_image = generator(z).reshape(1, 28, 28)
            images.append(generated_image.clone())
    
    return z.detach(), images


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_gif_from_images(folder_path, output_path, duration=100):
    """
    Crée un GIF à partir des images dans un dossier.
    
    Args:
        folder_path (str): Chemin du dossier contenant les images.
        output_path (str): Chemin du fichier GIF à créer.
        duration (int): Durée d'affichage de chaque image dans le GIF (en millisecondes).
    """
    # Obtenir une liste triée des fichiers d'images dans le dossier
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    
    # Charger toutes les images
    images = [Image.open(image_file) for image_file in image_files]
    
    # Créer et sauvegarder le GIF
    if images:
        images[0].save(
            output_path, 
            save_all=True, 
            append_images=images[1:], 
            duration=duration, 
            loop=0
        )
        print(f"GIF créé et sauvegardé sous : {output_path}")
    else:
        print("Aucune image trouvée pour créer le GIF.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Latent Space Optimal Transport GIF.')
    parser.add_argument("--batch_size", type=int, default=2048, help="The batch size to use for training.")
    parser.add_argument("--OT", type=str2bool, default=True, help="Whether to apply optimal transport.")            
    args = parser.parse_args()

    print('Model Loading...')
    mnist_dim = 784

    # Charger les modèles
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model_gene(model, 'checkpoints', 201)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    discriminator = Discriminator(mnist_dim).to(device)
    discriminator = load_model_discr(discriminator, 'checkpoints', 201)
    discriminator.eval()
    print('Model loaded.')

    folder_latent = 'samples_latent'
    clean_folder(folder_latent)

    print("Utilisation du transport optimal dans l'espace latent")
    z = torch.randn(1, 100).to(device)

    # Calcul de K_eff pour l'espace latent
    keff_latent = 0.47  # Remplacer par `calculate_keff` si nécessaire.

    # Appliquer le transport optimal dans l'espace latent
    z_final, images_latent = latent_space_dot_gif(model, discriminator, z, keff_latent)

    # Sauvegarder les images générées pour chaque étape
    for k in range(len(images_latent)):
        torchvision.utils.save_image(images_latent[k], os.path.join(folder_latent, f'{k}.png'))

    # Créer le GIF à partir des images sauvegardées
    output_gif_latent = 'output_latent.gif'
    create_gif_from_images(folder_latent, output_gif_latent)
