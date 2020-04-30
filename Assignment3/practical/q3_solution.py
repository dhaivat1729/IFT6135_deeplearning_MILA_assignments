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
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)
    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # COMPLETE TRAINING PROCEDURE

    for iter_val in range(n_iter):
        fake_latent_vars = torch.randn(train_batch_size, z_dim, device = device, requires_grad=True).to(device)
        fake_images = generator(fake_latent_vars).detach().to(device)
        real_images, _ = next(iter(train_loader))
        real_images = real_images.to(device)
        optim_critic.zero_grad()
        ## wasserstein loss time 
        px = critic(fake_images)
        py = critic(real_images)

        
        
        ## wasserstein distance
        wasserstein_dist = (torch.mean(px)-torch.mean(py))
        
        ## lipschitz penalty
        a = torch.rand(fake_images.size()[0], 1, device=device).unsqueeze(-1).unsqueeze(-1)
        a = a.expand(fake_images.size()).to(device)
        z = fake_images*a + real_images*(1 - a)
        z = torch.autograd.Variable(z,requires_grad=True)
        fz = critic(z)
        
        grad_z = torch.autograd.grad(outputs=fz, inputs=z,
                                   grad_outputs=torch.ones(fz.size()).to(device),
                                   create_graph=True, retain_graph=True)[0]
        grad_z = grad_z.view(grad_z.size(0),-1).to(device)
        out = torch.mean(torch.relu(torch.norm(grad_z,p=2,dim=-1, keepdim=True)-1)**2,dim=0).to(device)
        lp = lp_coeff * out

        loss_D = lp + wasserstein_dist
        print("Iter is: ", iter_val, "  Loss discriminator: ", loss_D.item())
        ## backward pass
        loss_D.backward()

        optim_critic.step()

        if iter_val % n_critic_updates == 0:
            optim_generator.zero_grad()
            gen_images = generator(fake_latent_vars)
            loss_G = -torch.mean(critic(gen_images.to(device)))
            print("Loss generator: ", loss_G.item())
            loss_G.backward()
            optim_generator.step()

        if iter_val % 10000 == 0:
            torch.save(generator.state_dict(), 'generator_' + str(iter_val).zfill(10) + '.model')
            torch.save(critic.state_dict(), 'critic_' + str(iter_val).zfill(10) + '.model')
            fake_latent_vars = torch.randn(100, z_dim, device = device, requires_grad=True).to(device)
            fake_images = generator(fake_latent_vars).detach().to(device)
            torchvision.utils.save_image(fake_images.data, "temp%d.png" % iter_val, normalize=True) 

    # COMPLETE QUALITATIVE EVALUATION
import ipdb; ipdb.set_trace()