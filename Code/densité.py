import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from lenet import LeNet5  # Assurez-vous que ce fichier est disponible dans votre environnement
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Resize

# Transformation pour MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisation
])

# Chargement des données MNIST
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Création d'un DataLoader
mnist_loader = DataLoader(mnist_test, batch_size=len(mnist_test), shuffle=False)

# Extraction des données et labels MNIST
mnist_data, mnist_labels = next(iter(mnist_loader))
mnist_data = mnist_data.numpy().reshape(len(mnist_test), -1)  # Flatten MNIST images
mnist_labels = mnist_labels.numpy()

# Définir le chemin vers le dossier contenant les images
samples_folder = "samples"  # Remplacez par le chemin réel

# Transformation des images : redimensionner à 32x32 et convertir en tenseur
transform = Compose([
    Resize((32, 32)),
    ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Charger toutes les images et les convertir en tenseur
samples_data = []
for file_name in os.listdir(samples_folder):
    if file_name.endswith('.png'):  # Assurez-vous de ne charger que les fichiers PNG
        img_path = os.path.join(samples_folder, file_name)
        img = Image.open(img_path).convert('L')  # Convertir en niveau de gris
        img_tensor = transform(img)  # Appliquer les transformations
        samples_data.append(img_tensor.numpy())

samples_data = np.array(samples_data)  # Convertir en tableau numpy

# Charger le modèle
model = LeNet5()
model.load_state_dict(torch.load('lenet_epoch=12_test_acc=0.991.pth'))
model.eval()  # Mettre le modèle en mode évaluation

# Redimensionner les données pour les adapter à LeNet5
samples_data_reshaped = samples_data.reshape(-1, 1, 32, 32)  # Format (Batch, Channel, Height, Width)
samples_data_tensor = torch.tensor(samples_data_reshaped, dtype=torch.float32)

# Prédictions
with torch.no_grad():
    outputs = model(samples_data_tensor)
    samples_labels = outputs.argmax(dim=1).numpy()

# PCA pour MNIST
pca = PCA(n_components=1)
mnist_pca_data = pca.fit_transform(mnist_data)  # Une seule composante principale
mnist_pca_df = pd.DataFrame({
    '1st_principal': mnist_pca_data[:, 0],
    'labels': mnist_labels
})

# PCA pour `samples`
samples_data_flattened = samples_data_reshaped.reshape(samples_data_reshaped.shape[0], -1)
samples_pca_data = pca.fit_transform(samples_data_flattened)  # Une seule composante principale
samples_pca_df = pd.DataFrame({
    '1st_principal': samples_pca_data[:, 0],
    'labels': samples_labels
})

# Visualisation des densités
plt.figure(figsize=(10, 6))

# Densité pour MNIST
sns.kdeplot(
    mnist_pca_df['1st_principal'], label='MNIST', fill=True, alpha=0.6, color="blue"
)

# Densité pour Samples
sns.kdeplot(
    samples_pca_df['1st_principal'], label='Samples', fill=True, alpha=0.6, color="green"
)

# Configuration de l'intrigue
plt.title("Density Plot of 1st Principal Component (PCA)")
plt.xlabel("1st Principal Component")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

