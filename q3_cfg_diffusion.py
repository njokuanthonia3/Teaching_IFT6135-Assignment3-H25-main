# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        u = t.float() / (self.n_steps)
        b = torch.atan(torch.exp(torch.tensor(-self.lambda_max / 2))) 
        a = torch.atan(torch.exp(torch.tensor(-self.lambda_min / 2))) - b
        lambda_t = -2 * torch.log(torch.tan(a * u + b))
        lambda_t = lambda_t.reshape(-1, 1, 1, 1)
        return lambda_t
    
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        # exp_lambda = torch.exp(lambda_t)
        # var = exp_lambda / (1 + exp_lambda)
        # return var.sqrt()
        exp_lambda = torch.exp(-lambda_t)
        var = 1 / (1 + exp_lambda)
        return var.sqrt()
    
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        # exp_lambda = torch.exp(lambda_t)
        # var= 1 / (1 + exp_lambda)
        # return var.sqrt()
        var = 1 - self.alpha_lambda(lambda_t)**2
        return var.sqrt()
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        alpha = self.alpha_lambda(lambda_t)
        sigma = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha * x + sigma * noise
        return z_lambda_t
    
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        var_q = (1 - (self.get_exp_ratio(lambda_t, lambda_t_prim)))* self.sigma_lambda(lambda_t)**2
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        var_q_x = (1 - (self.get_exp_ratio(lambda_t, lambda_t_prim)))* self.sigma_lambda(lambda_t_prim)**2
        # exp_ratio = torch.exp(lambda_t - lambda_t_prim)
        # var_q_x = (1 - exp_ratio) * torch.exp(-lambda_t_prim)
        # var_q_x = var_q_x.clamp(min=1e-10)  # prevent instability
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
    
        #Compute alpha and sigma values
        alpha_t = self.alpha_lambda(lambda_t)
        alpha_t_prim = self.alpha_lambda(lambda_t_prim)
        sigma_t = self.sigma_lambda(lambda_t)
        sigma_t_prim = self.sigma_lambda(lambda_t_prim)

        # Compute the scaling factors
        # scale1 = alpha_t_prim / alpha_t
        # scale2 = (sigma_t_prim / sigma_t) ** 2
        coeff_z = sigma_t_prim / sigma_t
        coeff_x = alpha_t_prim - coeff_z * alpha_t

        # Compute the mean
        mu = coeff_z * z_lambda_t + coeff_x * x
        return mu
        

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        sigma_t = self.sigma_lambda(lambda_t)
        sigma_t_prim = self.sigma_lambda(lambda_t_prim)
        var = (sigma_t_prim / sigma_t) ** 2 * v
        #var = var.clamp(min=1e-10)  # prevent instability
        return var

    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        noise = torch.randn_like(mu)
        sample = mu + torch.sqrt(var) * noise
        return sample 


    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        #TODO: q_sample z
        lambda_t = self.get_lambda(t)
        # Forward process: sample z_lambda_t using q_sample
        z_lambda_t = self.q_sample(x0, lambda_t, noise)
        # Predict the noise using the epsilon model
        eps_theta = self.eps_model(z_lambda_t, labels)
        # Compute the MSE loss between the predicted noise and the actual noise
        loss = F.mse_loss(eps_theta, noise)
        return loss









