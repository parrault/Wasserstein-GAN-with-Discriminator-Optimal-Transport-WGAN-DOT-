import os
from tqdm import trange
import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
from utils import save_models, device, visualize_metrics, calculate_Keff, load_model_discr, gradient_penalty,load_model_gene, calculate_keff


def train_wgan(args):
    # Hyperparameters
    CRITIC_ITERATIONS = 5
    Lambda_gp = 10
    mnist_dim = 784  # MNIST images are 28x28
    z_dim = 100  # Dimension of the noise vector
    channel_img = 1

    # Create directories for checkpoints and generated images
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)

    # Fixed vector for visualization
    z_fixed = torch.randn(12, z_dim).to(device)  # Fixed latent vector

    # Data Pipeline
    print('Loading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    print('Dataset loaded.')

    # Model Initialization
    print('Initializing models...')
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)
    print('Models initialized.')

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

 
    G = load_model_gene(G, 'checkpoints', 201)
    D = load_model_discr(D, 'checkpoints', 201)
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    

    # Training Loop
    print('Starting training...')
    G_losses, D_losses, keff_values = [], [], []

    for epoch in trange(1, args.epochs + 1, leave=True):
        G.train()
        D.train()
        G_loss_epoch, D_loss_epoch, n_samples = 0, 0, 0

        if epoch > 40 : CRITIC_ITERATIONS = 1

        for x, _ in train_loader:
            x = x.view(-1, mnist_dim).to(device)

            # Train Discriminator (Critic)
            for _ in range(CRITIC_ITERATIONS):
                z = torch.randn(x.size(0), z_dim).to(device)
                fake_data = G(z).detach()
                D_real = D(x)
                D_fake = D(fake_data)
                real_loss = -torch.mean(D_real)
                fake_loss = torch.mean(D_fake)
                gp = gradient_penalty(D, x, fake_data, device=device)
                d_loss = real_loss + fake_loss + Lambda_gp * gp

                D_optimizer.zero_grad()
                d_loss.backward()
                D_optimizer.step()

            # Train Generator
            z = torch.randn(x.size(0), z_dim).to(device)
            fake_data = G(z)
            g_loss = -torch.mean(D(fake_data))

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Accumulate losses
            n_samples += x.size(0)
            D_loss_epoch += d_loss.item() * x.size(0)
            G_loss_epoch += g_loss.item() * x.size(0)

        # Record average losses
        D_losses.append(D_loss_epoch / n_samples)
        G_losses.append(G_loss_epoch / n_samples)

        # Generate and save images using the fixed vector
        save_models(G, D, 'checkpoints', epoch+201)
        with torch.no_grad():
            generator = load_model_gene(Generator(g_output_dim = mnist_dim).to(device), 'checkpoints',epoch+201 )
            generator.eval()
            generated_images = generator(z_fixed).view(12, 1, 28, 28)  # Reshape to image format
            save_image(generated_images, f'generated_images/epoch_{epoch+201:03d}.png', nrow=4, normalize=True)

        
        discriminator = load_model_discr(Discriminator(mnist_dim).to(device), 'checkpoints', epoch+201)
        discriminator.eval()
      
        z = torch.randn(5000, z_dim).to(device)
        generated_samples = G(z).to('cpu')
        K_eff = calculate_Keff(discriminator, generated_samples, num_pairs=500)
       
        keff_values.append((epoch, K_eff))

        print(f"Epoch {epoch}: D_loss = {D_losses[-1]:.4f}, G_loss = {G_losses[-1]:.4f}, K_eff = {K_eff:.4f}")

    # Visualize training metrics
    print('Training completed. Visualizing metrics...')
    visualize_metrics(D_losses, G_losses, keff_values, save_path='metrics_visualization.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Wasserstein GAN with Gradient Penalty (WGAN-GP).')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    args = parser.parse_args()

    train_wgan(args)
