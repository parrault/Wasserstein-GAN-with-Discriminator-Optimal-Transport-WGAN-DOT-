import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lenet import LeNet5  # Assurez-vous que votre modèle LeNet5 est correctement importé
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
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
def calculate_precision_recall_distance(real_embeddings, fake_embeddings, k=5, metric='euclidean'):
    """
    Calcule la précision et le rappel avec une approche locale en utilisant les k plus proches voisins.
    """
    from sklearn.neighbors import NearestNeighbors

    if metric not in ['euclidean', 'cosine']:
        raise ValueError("Metric should be 'euclidean' or 'cosine'.")

    # Initialiser les k-NN pour les deux ensembles
    knn_real = NearestNeighbors(n_neighbors=k, metric=metric).fit(real_embeddings)
    knn_fake = NearestNeighbors(n_neighbors=k, metric=metric).fit(fake_embeddings)

    # Précision : Les faux qui ont au moins un vrai proche
    distances_fake_to_real, _ = knn_real.kneighbors(fake_embeddings)
    precision = np.mean(distances_fake_to_real[:, -1] <= np.median(distances_fake_to_real))

    # Rappel : Les vrais qui ont au moins un faux proche
    distances_real_to_fake, _ = knn_fake.kneighbors(real_embeddings)
    recall = np.mean(distances_real_to_fake[:, -1] <= np.median(distances_real_to_fake))

    return precision, recall


# Prédire les chiffres pour les images générées
def predict_generated_digits(model, dataloader, device):
    predicted_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Obtenez les prédictions de classe
            predicted_labels.extend(predicted.cpu().numpy())  # Ajouter les prédictions à la liste
    return np.array(predicted_labels)

# Calculer la proportion de chaque chiffre généré
def calculate_digit_proportions(predicted_labels):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    proportions = dict(zip(unique, counts / len(predicted_labels)))
    return proportions

# Afficher un graphique de la proportion des chiffres
def plot_digit_proportions(proportions):
    import matplotlib as mpl  # Assurez-vous d'avoir cette importation

    # Extraire les chiffres et leurs proportions
    digits = list(proportions.keys())
    values = [proportions[digit] for digit in digits]
    
    # Configurer le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Paired.colors[:len(digits)]
    bars = ax.bar(digits, values, color=colors, edgecolor='black')
    
    # Ajouter des annotations
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2%}",
            ha='center', fontsize=10, fontweight='bold'
        )
    
    # Personnaliser le design
    ax.set_xlabel('Chiffres générés', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des chiffres générés', fontsize=14, fontweight='bold')
    ax.set_xticks(digits)
    ax.set_xticklabels(digits, fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(values) + 0.1)  # Ajouter une marge supérieure pour l'annotation
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter une colorbar avec ScalarMappable
    norm = mpl.colors.Normalize(vmin=0, vmax=len(digits) - 1)
    sm = mpl.cm.ScalarMappable(cmap='Paired', norm=norm)
    sm.set_array([])  # Nécessaire pour éviter l'erreur

    # Spécifiez explicitement un axe pour la colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label("Couleurs associées aux chiffres", fontsize=12)
    cbar.set_ticks(range(len(digits)))
    cbar.set_ticklabels(digits)

    plt.tight_layout()
    plt.show()





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

    # Prédire les chiffres pour les images générées
    print("Prédiction des chiffres générés...")
    predicted_labels = predict_generated_digits(lenet, generated_dataloader, device)

    # Calculer et afficher la proportion de chaque chiffre
    proportions = calculate_digit_proportions(predicted_labels)
    print("Proportion de chaque chiffre généré :")
    for digit, proportion in proportions.items():
        print(f"Chiffre {digit}: {proportion * 100:.2f}%")

    # Afficher le graphique des proportions
    plot_digit_proportions(proportions)

    # Calcul du FID
    print("Calcul du FID...")
    fid_score = calculate_fid(real_embeddings, fake_embeddings)
    print(f"FID Score: {fid_score:.4f}")

    # Calcul précision et rappel
    print("Calcul de la précision et du rappel avec k-NN...")
    precision, recall = calculate_precision_recall_distance(real_embeddings, fake_embeddings)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    
    

