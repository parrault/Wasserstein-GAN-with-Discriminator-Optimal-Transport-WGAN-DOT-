import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lenet import LeNet5
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Resize
import seaborn as sns
from model import Generator, Discriminator
from utils import device, load_model_discr, load_model_gene

# Configuration
mnist_dim = 784

# Chargement du générateur et du discriminateur
generator = Generator(g_output_dim=mnist_dim).to(device)
generator = load_model_gene(generator, 'checkpoints', 201)
generator = torch.nn.DataParallel(generator).to(device)
generator.eval()

discriminator = Discriminator(mnist_dim).to(device)
discriminator = load_model_discr(discriminator, 'checkpoints', 201)
discriminator.eval()

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

# Génération et projection du point unique z
z = torch.randn(1, 100).to(device)  # Générer un vecteur latent
z_generated = generator(z).detach()  # Générer une donnée unique

# Vérifiez la sortie et redimensionnez si nécessaire
print("Shape of z_generated before reshape:", z_generated.shape)

# Reformatez en [batch_size, channels, height, width] si nécessaire
if z_generated.shape[1] == 32 * 32:  # Si aplatie
    z_generated = z_generated.view(-1, 1, 32, 32)  # Reformate en image
elif z_generated.shape[1] == 28 * 28:  # Si sortie en 28x28
    z_generated = z_generated.view(-1, 1, 28, 28)
    z_generated = torch.nn.functional.interpolate(z_generated, size=(32, 32), mode='bilinear', align_corners=False)

print("Shape of z_generated after reshape:", z_generated.shape)

# Aplatir pour PCA
z_generated_flat = z_generated.view(1, -1)  # Aplatir pour PCA
z_pca_data = pca.transform(z_generated_flat.cpu().numpy())  # Projeter dans l'espace PCA

z_pca_df = pd.DataFrame({
    '1st_principal': [z_pca_data[0, 0]],
    '2nd_principal': [z_pca_data[0, 1]],
    'labels': ['z']  # Label distinct pour z
})

# Combiner les données générées avec le point unique z
combined_pca_df = pd.concat([samples_pca_df, z_pca_df])

# Visualisation
plt.figure(figsize=(10, 6))

# Tracer les données générées
sns.scatterplot(
    x='1st_principal',
    y='2nd_principal',
    hue='labels',
    palette='tab10',
    data=samples_pca_df,
    alpha=0.6,
    s=50
)

# Ajouter le point z
plt.scatter(
    z_pca_df['1st_principal'],
    z_pca_df['2nd_principal'],
    color='red',
    s=200,
    label='z'
)

# Configurer le graphique
plt.title("PCA Visualization: Generated Data with Highlighted Latent Point z")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.legend(title="Labels", loc='upper right')
plt.tight_layout()
plt.show()

