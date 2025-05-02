"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

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

    #TODO: compute log_likelihood_bernoulli
    ll_bernoulli = target * torch.log(mu + 1e-8) + (1 - target) * torch.log(1 - mu + 1e-8)
    ll_bernoulli = ll_bernoulli.sum(dim=1)  # sum over input_size

    return ll_bernoulli


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

    #TODO: compute log normal
    ll_normal = -0.5 * (logvar + ((z - mu) ** 2) / torch.exp(logvar) + math.log(2 * math.pi))
    ll_normal = ll_normal.sum(dim=1)  # sum over input_size
    
    return ll_normal


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

    #TODO: compute log_mean_exp
    a_i = torch.max(y, dim=1, keepdim=True)[0]
    lme = a_i + torch.log(torch.mean(torch.exp(y - a_i), dim=1, keepdim=True))
    lme = lme.squeeze(1)  # remove the second dimension

    return lme 


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

    #TODO: compute kld
    kl_gg = 0.5 * (
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1
    )
    kl_gg = kl_gg.sum(dim=1)  # sum over input_size

    return kl_gg


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
    batch_size = mu_q.size(0)
    input_size = mu_q.size(1)
    
    # Reshaping and expanding for sampling
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # Standard deviation
    std_q = torch.exp(0.5 * logvar_q)
    
    # Sample z ~ q(z)
    eps = torch.randn_like(std_q)  # noise
    z = mu_q + eps * std_q

    # Log probability of z under q
    log_qz = -0.5 * (logvar_q + torch.pow((z - mu_q) / std_q, 2) + torch.log(torch.tensor(2 * torch.pi)))

    # Log probability of z under p
    std_p = torch.exp(0.5 * logvar_p)
    log_pz = -0.5 * (logvar_p + torch.pow((z - mu_p) / std_p, 2) + torch.log(torch.tensor(2 * torch.pi)))

    # Sum over dimensions
    log_qz = log_qz.sum(dim=-1)
    log_pz = log_pz.sum(dim=-1)

    # KL per sample
    kl_per_sample = log_qz - log_pz  # (batch_size, num_samples)

    # Monte Carlo estimate: mean over samples
    kl_mc = kl_per_sample.mean(dim=1)  

    return kl_mc

