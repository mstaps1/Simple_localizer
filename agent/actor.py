import torch
import torch.nn as nn
import numpy as np




class SquashedGaussianMLPActor(nn.Module):
  def __init__(self, beta, env_params, act_limit):
    # Spinning up algorithm
    super().__init__()
    self.max_action = env_params['action_max']
    self.input_dims_1 = env_params['obs1']
    self.input_dims_2 = env_params['obs2']
    
    self.hidden_1_size = 256
    self.hidden_1_num_layers = 4
    self.hidden_1_out = 30
    
    # Set up NN to handle obs 1 input
    self.Linear_block_1 = MLP(self.input_dims_1,
                         self.hidden_1_size,
                         self.hidden_1_out,
                         self.hidden_1_num_layers)
    #Set up NN to handle obs 2 input
    N = 1
    Channels = 1 #single Channel
    Channels_in = (N, Channels, self.input_dims_2[0], self.input_dims_2[1])
    Channels_1 = (N, Channels, 30, 30)
    channels_out = (N, Channels, 10, 10)

    # Start out with equal stride and kernel change possibly.
    kernel_size, stride = 3, 2

    #Conv blocks for obs2
    self.conv_blocks1 = self.conv_block(Channels, Channels, kernel_size, stride)
    self.conv_blocks2 = self.conv_block(Channels,Channels,kernel_size, stride)
    self.conv_blocks3 = self.conv_block(Channels, Channels, kernel_size, stride)    
    
    
    self.input_dim_2 = 1182
    
    # Set up linear layers for output
    self.hidden_2_size = 256
    self.hidden_2_num_layers = 4
    self.outsize_2 = 128

    # linear block taking in all 3 inputs
    self.Linear_block_2 = MLP(self.input_dim_2,
                         self.hidden_2_size,
                         self.outsize_2,
                         self.hidden_2_num_layers)
    
    # 
    self.mu_layer = nn.Linear(self.outsize_2[-1], act_dim)
    self.log_std_layer = nn.Linear(self.outsize_2[-1], act_dim)
    self.act_limit = act_limit
    
    
  def forward(self, obs, EvalMode=False):
    net_out = self.net(obs)
    mu = self.mu_layer(net_out)
    log_std = self.log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    
    #Pre-squash distribution and sample
    pi_distribution = Normal(mu, std)
    # Sample from the distribution
    if EvalMode:
      pi_action = mu
    else:
      pi_action = pi_distribution.rsample()
    #Squash
    pi_action = torch.tanh(pi_action)
    #go back to limit 
    pi_action = self.act_limit * pi_action

    return pi_action