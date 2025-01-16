import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, device, calculate_keff, visualize_metrics ,load_model_discr, load_model_gene




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    model = Generator(g_output_dim = mnist_dim).to(device)
    model = load_model_gene(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    discriminator = Discriminator(mnist_dim).to(device)
    discriminator = load_model_discr(discriminator, 'checkpoints')
    discriminator.eval()


    print('Start Training :')
    
    n_epoch = args.epochs
    G_losses = []
    D_losses = []
    keff_values = []

    for epoch in trange(1, n_epoch+1, leave=True):  
        n_samples = 0
        G_loss_epoch = 0
        D_loss_epoch = 0         
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            d_loss = D_train(x, G, D, D_optimizer, criterion)
            
            g_loss = G_train(x, G, D, G_optimizer, criterion)
            
            n_samples += x.size(0)
            #n_samples += 1
            D_loss_epoch += d_loss * x.size(0)
            G_loss_epoch += g_loss* x.size(0)
            #D_loss_epoch += d_loss 
            #G_loss_epoch += g_loss
        D_losses.append(D_loss_epoch / n_samples )
        G_losses.append(G_loss_epoch /n_samples)

        if epoch%1 == 0:
            save_models(G, D, 'checkpoints')
            with torch.no_grad():
                discriminator = Discriminator(mnist_dim).to(device)
                discriminator = load_model_discr(discriminator, 'checkpoints')
                discriminator.eval()
                # Générer directement les échantillons nécessaires pour K_eff
                z = torch.randn(5000, 100).to(device)
                generated_samples = G(z).to('cpu')
                # Calcul de K_eff
                K_eff = calculate_keff(discriminator, generated_samples, num_pairs=500)
                keff_values.append((epoch, K_eff))
            print(f"Epoch {epoch}: D_loss = {D_losses[-1]:.4f}, G_loss = {G_losses[-1]:.4f}, K_eff = {K_eff:.4f}")
        
    print('Training done')
    print('Training done. Visualizing metrics...')
    visualize_metrics(D_losses, G_losses, keff_values, save_path='metrics_visualization.png')

        
