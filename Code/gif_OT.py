import torch 
import torchvision
import os
import argparse
import shutil
from PIL import Image
from model import Generator, Discriminator
from utils import load_model_gene, device, load_model_discr, latent_space_dot


def clean_folder(folder_path):
    """
    Supprime tous les fichiers dans un dossier.
    
    Args:
        folder_path (str): Chemin du dossier à nettoyer.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Supprime le dossier et son contenu
    os.makedirs(folder_path, exist_ok=True)  # Recrée un dossier vide


def target_space_dot_gif(discriminator, y, K_eff, epsilon = 0.5, delta=1e-3, n_updates = 14):
    # Initialisation
    x = y.clone().detach().requires_grad_(True).to(y.device)
    optimizer = torch.optim.Adam([x], lr=epsilon)  # Utiliser Adam pour optimiser x
    #ici on veut minimiser la fonction x-y+delta -(1/k)*D(x)
    images = []
    for _ in range(n_updates):
        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Calcul de la sortie du discriminateur
        D_x = discriminator(x).view(-1)

        # Calcul de l'objectif
        norm_term = torch.norm(x - y + delta, p=2, dim=1)  # ||x - y + delta||_2
        objective = norm_term - (1 / K_eff) * D_x

        # Calcul du gradient et optimisation
        loss = objective.mean()  # Moyenne des objectifs pour les batches
        loss.backward(retain_graph=True)  # Calcul des gradients
        optimizer.step()  # Mise à jour des paramètres
        y = x.clone()
        images.append(y.reshape(1, 28, 28))
    return x.detach(), images

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--OT", type=str2bool, default=True,
                      help="The batch size to use for training.")            
    args = parser.parse_args()


    print(args.OT)
    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).to(device)
    model = load_model_gene(model, 'checkpoints', 201)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    discriminator = Discriminator(mnist_dim).to(device)
    discriminator = load_model_discr(discriminator, 'checkpoints', 201)
    discriminator.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    L = []
    #with torch.no_grad():

    folder_modify = 'samples_modify'
    
    # Nettoyer le dossier
    clean_folder(folder_modify)

    print("utilisation du transport optimal")
    z = torch.randn(1, 100).to(device)
    #z_transformed = latent_space_dot(model, discriminator, z, 0.47,  0.005, 30, delta=1e-3)
    x = model(z)
    y = x.reshape(1, 28, 28)
    images = [y]
    x, images_evol= target_space_dot_gif(discriminator, x, 0.52)#0.08 avant
    x = x.reshape(1, 28, 28)
    images += images_evol
    for k in range(len(images)):
        torchvision.utils.save_image(images[k], os.path.join('samples_modify', f'{k}.png'))         
          



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



    # Chemin de sortie du GIF
output_gif_path = 'output.gif'

    # Créer le GIF à partir des images générées
create_gif_from_images(folder_modify, output_gif_path)


