"""
Template for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    loss = (target*torch.log(mu) + (1-target)*torch.log(1-mu))
    out = torch.sum(loss, dim=-1)
    return out


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    out = torch.zeros(batch_size,)
    for i in range(batch_size):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i], torch.diag(logvar[i].exp()))
        out[i] = m.log_prob(z[i])
    return out


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)
    out = torch.zeros(batch_size,)

    for i in range(batch_size):
        yi_max = torch.max(y[i])
        yi = y[i] - yi_max
        out[i] = torch.log(torch.mean(torch.exp(yi))) + yi_max

    return out


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)
    out = torch.ones(batch_size,).type(torch.FloatTensor)

    for i in range(batch_size):
        sigma0 = torch.diag(logvar_q[i].exp())
        sigma1 = torch.diag(logvar_p[i].exp())
        mu0 = mu_q[i]
        mu1 = mu_p[i]

        ## find KL(N0||N1)
        ## https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        out[i] = (torch.trace(torch.matmul(torch.inverse(sigma1), sigma0)) + torch.matmul((mu1 - mu0), torch.matmul(torch.inverse(sigma1), (mu1 - mu0))) - len(mu1) + torch.log(torch.det(sigma1)/torch.det(sigma0)))/2.0

        # dq = torch.distributions.multivariate_normal.MultivariateNormal(mu_q[i], torch.diag(logvar_q[i].exp()))
        # dp = torch.distributions.multivariate_normal.MultivariateNormal(mu_p[i], torch.diag(logvar_p[i].exp()))
        # out[i] = torch.distributions.kl.kl_divergence(dq, dp)
    
    return out


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init

    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    out = torch.ones(batch_size,)
    out = out.type(torch.FloatTensor)

    
    for i in range(batch_size):
        dq = torch.distributions.multivariate_normal.MultivariateNormal(mu_q[i], torch.diag(logvar_q[i].exp()))
        dp = torch.distributions.multivariate_normal.MultivariateNormal(mu_p[i], torch.diag(logvar_p[i].exp()))
        kl_div = 0
        for j in range(num_samples):
            sample = dq.sample()
            kl_div += dq.log_prob(sample) - dp.log_prob(sample)
            
        out[i] = kl_div / float(num_samples)

    return out
