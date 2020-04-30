"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model
import numpy as np


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty

    """

    a = torch.from_numpy(next(iter(q2_sampler.distribution1(0, batch_size = x.size(0))))[:,1]).float()
    z = x*a[:,None] + y*(1 - a[:,None])
    z = torch.autograd.Variable(z,requires_grad=True)

    fz = critic(z)

    grad_z = torch.autograd.grad(outputs=fz, inputs=z,
                               grad_outputs=torch.ones(fz.size()),
                               create_graph=True, retain_graph=True)[0]

    grad_z = grad_z.view(grad_z.size(0),-1)

    out = torch.mean(torch.relu(torch.norm(grad_z,p=2,dim=-1, keepdim=True)-1)**2,dim=0)
    return out


def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    px = critic(x)
    py = critic(y)

    out = (torch.mean(px)-torch.mean(py))

    return out


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    px = 1.0 - torch.exp(-critic(x))
    py = 1.0 - torch.exp(-critic(y))
    fpy = -py/(1.0 - py)

    out = torch.mean(px) + torch.mean(fpy)
    return out


if __name__ == '__main__':
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.

    thetas = np.arange(0.0,2.1,0.1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    print("Using ", device)
    torch.manual_seed(10)
    np.random.seed(10)

    from matplotlib import pyplot as plt
    hellinger_dict = {}
    wd_dict = {}
    for theta in thetas:
        model1 = q2_model.Critic(2).to(device)
        optim1 = torch.optim.SGD(model1.parameters(), lr=1e-3)
        model2 = q2_model.Critic(2).to(device)
        optim2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
        
        lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.    
        iterations = 2500

        
        for i in range(iterations):
        
            # print("iteration and theta is: ", i, "  ", theta)

            ## data is the same for both the models
            sampler1 = iter(q2_sampler.distribution1(0, 512))
            sampler2 = iter(q2_sampler.distribution1(theta, 512))
            data1 = torch.from_numpy(next(sampler1)).to(device)
            data2 = torch.from_numpy(next(sampler2)).to(device)

            ## hellinger distance 
            
            out1 = model1(data1.type(dtype))
            out2 = model1(data2.type(dtype))

            ## let's compute the loss! 
            px = 1.0 - torch.exp(-out1)
            py = 1.0 - torch.exp(-out2)
            fpy = -py/(1.0 - py)
            loss_hellinger = -(torch.mean(px) + torch.mean(fpy))
            optim1.zero_grad()
            ## backward pass
            loss_hellinger.backward()

            ## optimizer step
            optim1.step()


            ## wasserstein with gradient penalty distance 
            
            out3 = model2(data1.type(dtype))
            out4 = model2(data2.type(dtype))

            ## let's compute the loss! 
            wd = torch.mean(out4)-torch.mean(out3)

            ## lipschitz penalty
            
            a = torch.from_numpy(next(iter(q2_sampler.distribution1(0, batch_size = data1.size(0))))[:,1]).to(device).float()
            z = data1*a[:,None] + data2*(1 - a[:,None])
            z = torch.autograd.Variable(z,requires_grad=True).to(device)
            
            fz = model2(z.type(dtype))

            grad_z = torch.autograd.grad(outputs=fz, inputs=z,
                                       grad_outputs=torch.ones(fz.size()).to(device),
                                       create_graph=True, retain_graph=True)[0]

            grad_z = grad_z.view(grad_z.size(0),-1)
            lipschitz_penalty = torch.mean(torch.relu(torch.norm(grad_z,p=2,dim=-1, keepdim=True)-1)**2)

            # print(lipschitz_penalty)
            loss_wasserstein = wd + lambda_reg_lp*lipschitz_penalty

            optim2.zero_grad()
            ## backward pass
            loss_wasserstein.backward()

            ## optimizer step
            optim2.step()

            # print("Iteration, theta, hellinger, wasserstein are: ", i, "  ", theta, " ", loss_hellinger.item(), " ", loss_wasserstein.item())

        print("training done!")
        data1 = torch.from_numpy(next(sampler1)).to(device)
        data2 = torch.from_numpy(next(sampler2)).to(device)
        
        ## hellinger compute
        out1 = model1(data1.type(dtype))
        out2 = model1(data2.type(dtype))

        ## let's compute the loss! 
        px = 1.0 - torch.exp(-out1)
        py = 1.0 - torch.exp(-out2)
        fpy = -py/(1.0 - py)
        loss_hellinger = (torch.mean(px) + torch.mean(fpy))

        ## storing it
        hellinger_dict[theta] = loss_hellinger.item()

        ## wd+lp
        out3 = model2(data1.type(dtype))
        out4 = model2(data2.type(dtype))

        ## let's compute the loss! 
        loss_wd = -(torch.mean(out4)-torch.mean(out3))

        ## storing it
        wd_dict[theta] = loss_wd.item()

        print("hellinger and wasserstein losses are: ", loss_hellinger.item(), " ", loss_wd.item())



    X = np.arange(0,2.1,0.1)
    Y = list(hellinger_dict.values())
    plt.plot(X,Y)
    plt.title('Square Hellinger Distance')
    plt.xlabel('theta')
    plt.ylabel('Distance')
    plt.show()
    plt.savefig('squared_hellinger.png')
    plt.clf()
    X = np.arange(0,2.1,0.1)
    Y = list(wd_dict.values())
    plt.plot(X,Y)
    plt.title('Wasserstein Distance')
    plt.xlabel('theta')
    plt.ylabel('Distance')
    plt.show()
    plt.savefig('wasserstein_dist.png')