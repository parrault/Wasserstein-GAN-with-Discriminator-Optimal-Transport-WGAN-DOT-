import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import torch
from model import Generator, Discriminator
from utils import load_model_discr, load_model_gene, device


def visualize_z_on_distribution(generator, discriminator, z, keff, epsilon=0.05, n_updates=100, delta=1e-3):
    """
    Visualise l'évolution de z se déplaçant sur la distribution de données générées G(z).

    Args:
        generator (torch.nn.Module): Générateur pré-entraîné.
        discriminator (torch.nn.Module): Discriminateur pré-entraîné.
        z (torch.Tensor): Échantillon latent initial.
        keff (float): Constante Lipschitz approximée.
        epsilon (float): Taux d'apprentissage pour la descente de gradient.
        n_updates (int): Nombre d'itérations de mise à jour.
        delta (float): Terme stabilisateur pour éviter les instabilités numériques.
    """
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=epsilon)

    # Stocker l'évolution de z et des distributions G(z)
    z_snapshots = []
    g_distributions = []

    for _ in range(n_updates):
        optimizer.zero_grad()

        term1 = torch.norm(z - z.detach() + delta, p=2) ** 2
        term2 = -1 / keff * discriminator(generator(z)).mean()
        loss = term1 + term2

        loss.backward()
        optimizer.step()
        z.data.clamp_(-1, 1)

        z_snapshots.append(z.clone().detach().cpu().numpy())  # Sauvegarde de z
        g_distributions.append(generator(z).clone().detach().cpu().numpy())  # Distribution de G(z)

    # Préparer les données pour l'animation
    z_snapshots = np.array(z_snapshots).flatten()  # Évolution de z
    g_distributions = np.array(g_distributions)  # Distribution des données générées

    # Animation
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        sns.kdeplot(g_distributions[frame], fill=True, alpha=0.5, color="blue", label="Distribution de G(z)")
        ax.scatter(z_snapshots[frame], 0, color="red", s=100, label=f"Point z (Itération {frame + 1})")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.1, 1)
        ax.set_title("Évolution de z sur la distribution de G(z)")
        ax.set_xlabel("Valeurs de G(z)")
        ax.set_ylabel("Densité")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(g_distributions), repeat=False)
    ani.save("z_on_distribution_evolution.gif", fps=10)
    plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    mnist_dim = 784

    # Charger le générateur et le discriminateur
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model_gene(model, 'checkpoints', 201)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    discriminator = Discriminator(mnist_dim).to(device)
    discriminator = load_model_discr(discriminator, 'checkpoints', 201)
    discriminator.eval()

    # Un seul échantillon latent initial
    z = torch.randn(1, 100, requires_grad=True).to(device)  # Espace latent initial de dim 100

    # Visualiser l'évolution pour un seul z
    visualize_z_on_distribution(model, discriminator, z, keff=0.47, epsilon=0.005, n_updates=30)


