import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lenet import LeNet5
from PIL import Image, ImageSequence
import os
import shutil  # Pour supprimer le contenu du dossier
from torchvision.transforms import Compose, ToTensor, Resize
import seaborn as sns
from model import Generator, Discriminator
from utils import device, load_model_discr, load_model_gene

# Configuration
mnist_dim = 784
output_folder = "z_mouvements"
keff = 0.47  # Constante Lipschitz approximée
epsilon = 2  # Taux d'apprentissage
n_updates = 30  # Nombre d'étapes de mise à jour
delta = 1e-3  # Terme stabilisateur

# Fonction pour vider le dossier
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Supprime tout le contenu du dossier
    os.makedirs(folder_path)  # Recrée un dossier vide

# Vider le dossier au début de l'exécution
clear_folder(output_folder)

# Chargement du générateur et du discriminateur
generator = Generator(g_output_dim=mnist_dim).to(device)
generator = load_model_gene(generator, 'checkpoints', 201)
generator = torch.nn.DataParallel(generator).to(device)
generator.eval()

discriminator = Discriminator(mnist_dim).to(device)
discriminator = load_model_discr(discriminator, 'checkpoints', 201)
discriminator.eval()
K_eff = 0.47
# Transformation des données
transform = Compose([
    Resize((32, 32)),
    ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Charger les images échantillons
samples_folder = "samples"  # Chemin réel
samples_data = []
for file_name in os.listdir(samples_folder):
    if file_name.endswith('.png'):
        img_path = os.path.join(samples_folder, file_name)
        img = Image.open(img_path).convert('L')
        img_tensor = transform(img)
        samples_data.append(img_tensor.numpy())

samples_data = np.array(samples_data)

# Chargement et prédictions avec LeNet5
lenet_model = LeNet5()
lenet_model.load_state_dict(torch.load('lenet_epoch=12_test_acc=0.991.pth'))
lenet_model.eval()

samples_data_reshaped = samples_data.reshape(-1, 1, 32, 32)
samples_data_tensor = torch.tensor(samples_data_reshaped, dtype=torch.float32)

with torch.no_grad():
    outputs = lenet_model(samples_data_tensor)
    samples_labels = outputs.argmax(dim=1).numpy()

# PCA pour les données générées
samples_data_flattened = samples_data_reshaped.reshape(samples_data_reshaped.shape[0], -1)
pca = PCA(n_components=2)
samples_pca_data = pca.fit_transform(samples_data_flattened)
samples_pca_df = pd.DataFrame({
    '1st_principal': samples_pca_data[:, 0],
    '2nd_principal': samples_pca_data[:, 1],
    'labels': samples_labels
})

# Initialisation de z
z_y = torch.randn(1, 100).to(device).requires_grad_(True)


# Génération des graphiques pour chaque étape
image_paths = []
# Génération des graphiques pour chaque étape (KDE sur 1ère composante principale)
image_paths = []
z = z_y.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=epsilon)
for step in range(n_updates):
    optimizer.zero_grad()
        
        # Calcul de l'objectif
    norm_term = torch.norm(z - z_y + delta, p=2)  # Norme ||z - z_y + delta||_2
    term1 = norm_term  # Première partie de l'objectif
    term2 = -1 / keff * discriminator(generator(z)).mean()  # Deuxième partie avec le discriminateur
    loss = term1 + term2

    # Backpropagation et optimisation
    loss.backward()
    optimizer.step()

        # Clamp les valeurs de z dans [-1, 1]
    z.data.clamp_(-1, 1)
    # Générer une image et mettre à jour z
    with torch.no_grad():
        z_generated = generator(z).detach()

        # Redimensionnement si nécessaire
        if z_generated.shape[1] == 32 * 32:
            z_generated = z_generated.view(-1, 1, 32, 32)
        elif z_generated.shape[1] == 28 * 28:
            z_generated = z_generated.view(-1, 1, 28, 28)
            z_generated = torch.nn.functional.interpolate(z_generated, size=(32, 32), mode='bilinear')

        # Aplatir pour PCA
        z_generated_flat = z_generated.view(1, -1)
        z_pca_data = pca.transform(z_generated_flat.cpu().numpy())

        z_pca_df = pd.DataFrame({
            '1st_principal': [z_pca_data[0, 0]],
            'labels': ['z']
        })

        # Estimation KDE et visualisation
        plt.figure(figsize=(10, 6))

        # KDE sur la 1ère composante principale
        sns.kdeplot(
            samples_pca_df['1st_principal'],
            fill=True,
            color="blue",
            alpha=0.6,
            label="KDE (Generated Data)"
        )

        # Ajouter le point z
        plt.scatter(
            z_pca_df['1st_principal'],
            [0],  # KDE est 0 pour un seul point
            color='red',
            s=200,
            label='z'
        )

        # Configurer le graphique
        plt.title(f"KDE on 1st Principal Component at Step {step}")
        plt.xlabel("1st Principal Component")
        plt.ylabel("Density")
        plt.legend(title="Legend", loc='upper right')
        plt.tight_layout()

        # Sauvegarder chaque étape
        image_path = os.path.join(output_folder, f"step_{step}.png")
        plt.savefig(image_path)
        image_paths.append(image_path)
        plt.close()

print(f"Graphiques sauvegardés dans le dossier : '{output_folder}'")

# Création du GIF
gif_path = os.path.join(output_folder, "latent_space.gif")
images = [Image.open(path) for path in image_paths]
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=300,  # Durée entre chaque frame en millisecondes
    loop=0
)

print(f"GIF généré : {gif_path}")
