"""
Template for Question 3.
@author: Samuel Lavoie
Edited by: Dhaivat Bhatt
"""
import torch
import torchvision
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator
import q2_sampler
from torch import optim
import numpy as np

if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 150000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 100
    train_batch_size_disentangled = 10
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    noise = 50
    np.random.seed(200)
    dims = np.arange(100)
    np.random.shuffle(dims)
    dims = dims[:12] 

    # train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)
    generator = Generator(z_dim=z_dim).to(device)
    generator.load_state_dict(torch.load('./generator_0000100000.model'))
    
    ## 0. Generative samples
    fake_latent_vars = torch.randn(train_batch_size, z_dim, device = device, requires_grad=True).to(device)
    fake_images = generator(fake_latent_vars).detach().to(device)
    torchvision.utils.save_image(fake_images.data, "sample_generated.png", nrow = 10, normalize=True) 
    

    ## 1. disentangled representation
    images_final = None
    fake_latent_vars = torch.randn(train_batch_size_disentangled, z_dim, device = device, requires_grad=True).to(device)
    images_final = generator(fake_latent_vars).detach().to(device)

    str_val = ''

    for dim_val in dims:

        str_val = str_val + '_' + str(dim_val)
        fake_latent_vars_new = fake_latent_vars.clone()
        fake_latent_vars_new[:,dim_val] += noise
        out_images = generator(fake_latent_vars_new).detach().to(device)
        images_final = torch.cat((images_final, out_images), dim = 0)

        # ## adding along each dimension(trying for first 64 dimensions)
        # for i in range(train_batch_size):
        #     fake_latent_vars[i][5] += noise_val
            
        # fake_images = generator(fake_latent_vars).detach().to(device)
    # import ipdb; ipdb.set_trace()
    torchvision.utils.save_image(images_final.data, "disentangled_representation{}.png".format(str_val), nrow = train_batch_size_disentangled, normalize=True) 
        

    ## 2.a

    alphas = np.arange(0,1.1,0.1)

    images_final = None
    for alpha in alphas:
        z0 = torch.randn(10, z_dim, device = device, requires_grad=True).to(device)        
        z1 = torch.randn(10, z_dim, device = device, requires_grad=True).to(device)        
        z_hat = alpha*z0 + (1 - alpha)*z1

        gen_images = generator(z_hat).detach().to(device)

        if images_final is not None:
            images_final = torch.cat((images_final, gen_images),0)
        else: 
            images_final = gen_images

    torchvision.utils.save_image(images_final.data, "interpolation_latent_space.png", nrow = 10, normalize=True) 


    ## 2.b
    alphas = np.arange(0,1.1,0.1)

    images_final = None
    for alpha in alphas:
        z0 = torch.randn(10, z_dim, device = device, requires_grad=True).to(device)        
        z1 = torch.randn(10, z_dim, device = device, requires_grad=True).to(device)        
        z0_images = generator(z0).detach().to(device)
        z1_images = generator(z1).detach().to(device)

        out_image = alpha*z0_images + (1 - alpha)*z1_images

        if images_final is not None:
            images_final = torch.cat((images_final, out_image),0)
        else: 
            images_final = out_image

    torchvision.utils.save_image(images_final.data, "interpolation_data_space.png", nrow = 10, normalize=True) 

    # COMPLETE QUALITATIVE EVALUATION
import ipdb; ipdb.set_trace()