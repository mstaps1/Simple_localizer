import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim

import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import *



class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, NN_size, hidden_size=32, init_w=3e-3, device='cpu' log_std_min=-20, log_std_max=20):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.device = device
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.sequence_length = NN_size['sequence_length']
        
        
        H_out = (H_in_conv + 2*padding-dilationx(Kernel_size-1))/stride+1
        
        
        
        