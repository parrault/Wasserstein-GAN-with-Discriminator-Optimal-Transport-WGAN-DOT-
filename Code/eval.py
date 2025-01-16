import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lenet import LeNet5  # Assurez-vous que votre modèle LeNet5 est correctement importé
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os

# Charger le modèle pré-entraîné LeNet5
def load_lenet(model_path, device):
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Transformations pour les images
def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convertir en grayscale
        transforms.Resize((32, 32)),                 # Redimensionner à 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# Charger les données MNIST réelles
def load_real_images(batch_size=200):
    mnist_dataset = MNIST(root='./data', train=True, download=True, transform=get_transforms())
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

# Charger les images générées avec leurs chemins
def load_generated_images(samples_dir, batch_size=200):
    image_paths = [
        os.path.join(samples_dir, fname)
        for fname in os.listdir(samples_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_paths:
        print("Aucune image trouvée dans le dossier.")
    
    print(f"Nombre d'images chargées : {len(image_paths)}")

    images = [get_transforms()(Image.open(path).convert("RGB")) for path in image_paths]
    
    # Vérifiez les transformations
    for i, img in enumerate(images[:5]):  # Afficher les 5 premières images
        print(f"Image {i + 1} transformée : {img.shape}")
    
    dataset = torch.utils.data.TensorDataset(torch.stack(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader, image_paths


# Afficher les images générées et leurs chemins
def display_fake_images(dataloader, image_paths, num_images=5):
    """
    Affiche un certain nombre d'images générées avec leurs chemins.

    Args:
        dataloader (DataLoader): DataLoader contenant les images.
        image_paths (list): Liste des chemins des images.
        num_images (int): Nombre d'images à afficher.
    """
    data_iter = iter(dataloader)
    batch = next(data_iter)  # Obtenir un batch
    images = batch[0][:num_images]  # Extraire les images des `num_images` premières

    # Afficher les chemins des images
    print("Chemins des 5 premières images générées :")
    for i, path in enumerate(image_paths[:num_images]):
        print(f"Image {i + 1}: {path}")

    # Afficher les images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img in enumerate(images):
        axes[i].imshow(img.squeeze(0).cpu().numpy(), cmap="gray")
        axes[i].axis("off")
    plt.show()

# Extraire les embeddings
def extract_embeddings(model, dataloader, device):
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            images = images.to(device)
            features = model.extract_features(images)
            embeddings.append(features.cpu().numpy())
    return np.vstack(embeddings)

# Calcul du FID
def calculate_fid(real_embeddings, fake_embeddings, epsilon=1e-6):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = fake_embeddings.mean(axis=0), np.cov(fake_embeddings, rowvar=False)
    
    # Ajouter une petite valeur à la diagonale pour éviter la singularité
    sigma1 += np.eye(sigma1.shape[0]) * epsilon
    sigma2 += np.eye(sigma2.shape[0]) * epsilon

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# Calcul de précision et rappel avec k-NN
def calculate_precision_recall(real_embeddings, fake_embeddings, threshold=0.5, metric='euclidean'):
   
    # Calculer les distances entre tous les embeddings
    if metric == 'euclidean':
        distances = euclidean_distances(fake_embeddings, real_embeddings)
    elif metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(fake_embeddings, real_embeddings)
        distances = 1 - similarities  # Convertir la similarité en distance
    else:
        raise ValueError("Metric must be 'euclidean' or 'cosine'.")

    # Précision : Fraction des embeddings générés ayant au moins un "match" réel
    matches_fake_to_real = np.any(distances <= threshold, axis=1)
    precision = np.mean(matches_fake_to_real)

    # Rappel : Fraction des embeddings réels ayant au moins un "match" généré
    matches_real_to_fake = np.any(distances.T <= threshold, axis=1)
    recall = np.mean(matches_real_to_fake)

    return precision, recall
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lenet_model_path = 'lenet_epoch=12_test_acc=0.991.pth'
    generated_images_dir = '/Users/p-arrow/Desktop/Cours/L3/Projet2/dl3project-2024-parrault/samples'

    # Charger le modèle
    print("Chargement du modèle LeNet5...")
    lenet = load_lenet(lenet_model_path, device)

    # Charger les images
    print("Chargement des images MNIST et générées...")
    real_dataloader = load_real_images()
    generated_dataloader, generated_image_paths = load_generated_images(generated_images_dir)

    # Afficher les 5 premières images générées
    print("Affichage de 5 images générées avec leurs chemins...")
    display_fake_images(generated_dataloader, generated_image_paths)

    # Extraction des embeddings
    print("Extraction des embeddings...")
    real_embeddings = extract_embeddings(lenet, real_dataloader, device)
    fake_embeddings = extract_embeddings(lenet, generated_dataloader, device)

    # Calcul du FID
    print("Calcul du FID...")
    fid_score = calculate_fid(real_embeddings, fake_embeddings)
    print(f"FID Score: {fid_score:.4f}")

    # Calcul précision et rappel
    print("Calcul de la précision et du rappel avec k-NN...")
    precision, recall = calculate_precision_recall(real_embeddings, fake_embeddings)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")





