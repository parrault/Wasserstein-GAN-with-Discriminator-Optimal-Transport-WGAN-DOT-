import torch
import os
import matplotlib.pyplot as plt


#Initialisation du device 

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Device used: {}".format(device))

#Fonction de train...:

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



# fonction pour save le D et le G ainsi que pour les load...................:

def save_models(G, D, folder, i):
    torch.save(G.state_dict(), os.path.join(folder,f'G{i}.pth'))
    torch.save(D.state_dict(), os.path.join(folder,f'D{i}.pth'))


def load_model_gene(G, folder, i):
    ckpt = torch.load(os.path.join(folder,f'G{i}.pth'), map_location=torch.device('cpu'),  weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_model_discr(D, folder, i):
    ckpt = torch.load(os.path.join(folder,f'D{i}.pth'), map_location=torch.device('cpu'),  weights_only=True)
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D



#Calcul de Keff
def calculate_Keff(discriminator, samples, num_pairs=20000):
     #on se place bien en mode eval
    discriminator.eval()
    N = samples.size(0)
    
    # Sélectionner des paires aléatoires, on génère 2 indices au hasard des samples, on fait ça pour 500.

    indices_x = torch.randint(0, N, (num_pairs,))
    indices_y = torch.randint(0, N, (num_pairs,))
    x = samples[indices_x].to(device)
    y = samples[indices_y].to(device)

    # sorties du discriminateur
    D_x = discriminator(x).view(-1)
    D_y = discriminator(y).view(-1)
    
    # Calcul des différences et des distances, d'après la fiche de recherche, on applique bien la norme
    distances = torch.norm(x - y, dim=1)  # Norme L2
    differences = torch.abs(D_x - D_y)   # difference en valeur absolue
    
    # Éviter la division par zéro
    distances[distances == 0] = 1e-8   # hyper important, j'ai eu des erreurs déjà
    
    # Calcul des ratios et récupération du maximum
    keff_values = differences / distances
    K_eff = keff_values.max().item()  #on prend la plus grand valeur
    
    return K_eff

def target_space_dot(discriminator, y, K_eff, epsilon = 0.08, delta=1e-3, n_updates = 13):
    """
    Implémente l'algorithme 1 du papier pour le transport optimal dans l'espace cible
    en utilisant un optimiseur.

    Args:
        discriminator (torch.nn.Module): Discriminateur entraîné.
        y (torch.Tensor): Échantillon initial généré (batch ou un seul point).
        K_eff (float): Constante de Lipschitz approximée.
        epsilon (float): Taux d'apprentissage pour la descente de gradient.
        delta (float): Terme stabilisateur pour éviter les instabilités numériques.
        n_updates (int): Nombre d'itérations de descente de gradient.

    Returns:
        torch.Tensor: Échantillon transporté après descente de gradient.
    """
    # Initialisation
    x = y.clone().detach().requires_grad_(True).to(y.device)
    optimizer = torch.optim.Adam([x], lr=epsilon)  # Utiliser Adam pour optimiser x
    #ici on veut minimiser la fonction x-y+delta -(1/k)*D(x)
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

    return x.detach() 


def visualize_metrics(D_losses, G_losses, keff_values, save_path='metrics_visualization.png'):
    """
    Visualise les pertes des modèles et l'évolution de K_eff, et sauvegarde le graphique.
    
    Arguments :
    - D_losses : liste des pertes du Discriminateur
    - G_losses : liste des pertes du Générateur
    - keff_values : liste des tuples (epoch, K_eff)
    - save_path : chemin pour sauvegarder l'image (par défaut 'metrics_visualization.png').
    """
    # Convertir keff_values en deux listes séparées pour l'affichage
    keff_epochs, keff_values = zip(*keff_values) if keff_values else ([], [])

    # Création des graphiques
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(D_losses, label='Discriminator Loss')
    plt.plot(G_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()

    # Plot de K_eff
    if keff_values:
        plt.subplot(1, 2, 2)
        plt.plot(keff_epochs, keff_values, marker='o', label='K_eff')
        plt.xlabel('Epoch')
        plt.ylabel('K_eff')
        plt.title('Evolution of K_eff')
        plt.legend()
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, 'No K_eff values to display', 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 fontsize=12, 
                 bbox=dict(facecolor='red', alpha=0.5))
        plt.axis('off')

    plt.tight_layout()
    # Sauvegarde du graphique
    if os.path.dirname(save_path):  # Si un répertoire est défini dans save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)  # Sauvegarde avant plt.show()
    print(f"Metrics visualization saved to {save_path}")

    # Affichage du graphique
    plt.show()

def gradient_penalty(critic, real, fake, device) :
    batch_size, C = real.shape
    epsilon = torch.rand(batch_size, 1).repeat(1, C).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(batch_size, -1)  
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty



import torch






def latent_space_dot(generator, discriminator, z_y, keff, epsilon=0.05, n_updates=100, delta=1e-3):
    """
    Applique le transport optimal dans l'espace latent et génère les étapes intermédiaires
    pour un GIF.

    Args:
        generator (torch.nn.Module): Générateur pré-entraîné.
        discriminator (torch.nn.Module): Discriminateur pré-entraîné.
        z_y (torch.Tensor): Échantillon latent cible.
        keff (float): Constante Lipschitz approximée.
        epsilon (float): Taux d'apprentissage pour la descente de gradient.
        n_updates (int): Nombre d'itérations de mise à jour.
        delta (float): Terme stabilisateur pour éviter les instabilités numériques.

    Returns:
        torch.Tensor: Échantillon latent après optimisation.
        list[torch.Tensor]: Liste des images générées pour chaque étape.
    """
    z = z_y.clone().detach().requires_grad_(True)  # Initialisation de z avec z_y
    optimizer = torch.optim.Adam([z], lr=epsilon)
    images = []

    for _ in range(n_updates):
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
        z.data.clamp_(-3, 3)

        # Sauvegarde des étapes pour visualisation
        #with torch.no_grad():
        #    images.append(generator(z).detach().cpu())

    return z.detach()


def calculate_keff(generator, discriminator, latent_samples, num_pairs=20000, device='cuda'):
    """
    Calcule l'approximation de la constante de Lipschitz (K_eff) pour un discriminateur 
    et un générateur donnés.

    Args:
        generator (torch.nn.Module): Le modèle générateur pré-entraîné.
        discriminator (torch.nn.Module): Le modèle discriminateur pré-entraîné.
        latent_samples (torch.Tensor): Les échantillons dans l'espace latent.
        num_pairs (int): Le nombre de paires aléatoires à utiliser.
        device (str): Le périphérique sur lequel effectuer les calculs ('cuda' ou 'cpu').

    Returns:
        float: La constante approximée K_eff.
    """
    # Mettre le discriminateur et le générateur en mode évaluation
    generator.eval()
    discriminator.eval()

    # Déplacer les données sur le bon appareil
    latent_samples = latent_samples.to(device)

    # Nombre total d'échantillons latents
    N = latent_samples.size(0)

    # Générer des indices aléatoires pour sélectionner des paires
    indices_x = torch.randint(0, N, (num_pairs,), device=device)
    indices_y = torch.randint(0, N, (num_pairs,), device=device)

    # Extraire les paires correspondantes dans l'espace latent
    z_x = latent_samples[indices_x]
    z_y = latent_samples[indices_y]

    # Générer les échantillons dans l'espace des données
    x = generator(z_x)
    y = generator(z_y)

    # Calculer les sorties du discriminateur
    with torch.no_grad():  # Désactiver le calcul des gradients
        D_x = discriminator(x).view(-1)
        D_y = discriminator(y).view(-1)

    # Calculer les distances et les différences
    distances = torch.norm(x - y, dim=1)  # Norme L2
    differences = torch.abs(D_x - D_y)   # Différence absolue des sorties

    # Éviter la division par zéro
    distances[distances == 0] = 1e-8

    # Calculer les ratios des différences sur les distances
    keff_values = differences / distances

    # Obtenir le maximum comme K_eff
    K_eff = keff_values.max().item()

    return K_eff

