import torch 
import torchvision
import os
import argparse


from model import Generator, Discriminator
from utils import load_model_gene, device, calculate_keff, load_model_discr, target_space_dot, calculate_Keff, latent_space_dot
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
    parser.add_argument("--K_calcul", type=bool, default=False,
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
    Ge =[]
    #with torch.no_grad():
    if  args.OT:
        print("utilisation du transport optimal")
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).to(device)
            z_transformed = latent_space_dot(model, discriminator, z, 0.47,  0.1 , 10, delta=1e-3) #0.005 , 30
            
                # Générer les échantillons dans l'espace des données
            x = model(z_transformed)
            Ge.append(z)
        
            L.append(x.to('cpu'))
            #x = target_space_dot(discriminator, x, 0.52, epsilon = 0.1, delta=1e-3, n_updates = 10)  #0.08 pour le gan
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1
            if n_samples%1000 == 0 :
                    print(f"{n_samples} générés........\n")
    else : 
        with torch.no_grad():
            while n_samples<10000:
                z = torch.randn(args.batch_size, 100).to(device)
            
                x = model(z)
                
                Ge.append(z)
                L.append(x.to('cpu'))
                
                x = x.reshape(args.batch_size, 28, 28)
                x = (x + 1) / 2
                x = torch.clamp(x, 0, 1)
                for k in range(x.shape[0]):
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'), normalize=False)
         
                        n_samples += 1
                if n_samples%2000 ==0 :
                    print(f"{n_samples} générés........\n")



    
    if args.K_calcul : 
        generated_samples = torch.cat(L, dim=0)
        latent_samples = torch.cat(Ge, dim=0)
        Kmax = 0
        for i in range(1):
            keff = calculate_keff(model, discriminator, latent_samples , num_pairs=50000, device=device)
            K  = calculate_Keff(discriminator,generated_samples, num_pairs = 50000 )
            
            if K > Kmax :
                Kmax = K
            if i%10 == 0 :  print("presque")
        print("K : ",K, "keff",  keff, sep = "\n")
    



