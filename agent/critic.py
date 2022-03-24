import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from utils import *


class CriticNetwork(nn.Module):
    def __init__(self, beta, env_params, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        """
        Critic network V0 (no reccurance), includes goal as input to be used with HER
        -----------------------------------------------------------------------------
        
        """
        
        self.max_action = env_params['action_max']
        self.input_dims_1 = env_params['obs1']
        self.input_dims_2 = env_params['obs2']

        self.input_dims_4 = np.shape(env_params['action_max'])[0]
        self.output_dims = 1
        
        self.hidden_1_size = 256
        self.hidden_1_num_layers = 4
        self.hidden_1_out = 30
        
        # Set up NN to handle obs 1 input
        self.Linear_block_1 = MLP(self.input_dims_1,
                             self.hidden_1_size,
                             self.hidden_1_out,
                             self.hidden_1_num_layers)
        #
        self.hidden_4_size = 256
        self.hidden_4_num_layers = 4
        self.hidden_4_out = 30
        
        self.Linear_block_4 = MLP(self.input_dims_4,
                             self.hidden_4_size,
                             self.hidden_4_out,
                             self.hidden_4_num_layers)
        #Set up NN to handle obs 2 input
        N = 1
        Channels = 1 #single Channel
        Channels_in = (N, Channels, self.input_dims_2[0], self.input_dims_2[1])
        Channels_1 = (N, Channels, 30, 30)
        channels_out = (N, Channels, 10, 10)
        
        # Start out with equal stride and kernel change possibly.
        kernel_size, stride = 3, 2
        
        #Conv blocks for obs2
        self.conv_blocks1 = self.conv_block(Channels, Channels, kernel_size,stride)
        self.conv_blocks2 = self.conv_block(Channels,Channels,kernel_size,stride)
        self.conv_blocks3 = self.conv_block(Channels, Channels, kernel_size,stride)
        
        
        # get 2x flattened conv_out size plus out_size of obs_1 mlp
        self.input_dim_2 = 1182
        
        
        # Set up linear layers for output
        self.hidden_2_size = 256
        self.hidden_2_num_layers = 4
        self.outsize_2 = 30
        
        # linear block taking in all 3 inputs
        self.Linear_block_2 = MLP(self.input_dim_2,
                             self.hidden_2_size,
                             self.outsize_2,
                             self.hidden_2_num_layers)
        
        # final linear layer
        self.Last_linear_block = nn.Sequential(nn.Linear(self.outsize_2, 1),
                                               nn.ReLU()
                                              )
        
        
        
        
    def linear_block(self,input_size, output_size):
        """
        Return a Linear Block with ReLu activation
        """
        return nn.Sequential(nn.Linear(input_size,output_size),
                                   nn.ReLU()
                            )
    def conv_block(self, input_channels, output_channels, kernel_size, stride):
        """
        convolution block with 2D convolution, batchnorm 2D, ReLU 
        """
        return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
        )
         
    def forward(self, obs, action):
        """
        Calculate the forwards pass
        -------------------------------
        Inputs: x dict with 'obs1' and 'obs2'
        obs 1 is 1D and includes pos_x, pos_y, con, Ux, Uy
        obs2 is 2D and is the Occupancy grid map
        goal is 2D and the goal Occupancy grid mad
        """
        
        # obs 1 processing
        out_1 = self.Linear_block_1(obs['obs1'])
        print("Critic out 1: ", out_1.shape)
        
        # Action processing
        print("Action Shape: ", action.shape)
        out_a = self.Linear_block_4(action)
        print(out_a.shape)
        
        # obs 2 processing
        out_2 = self.conv_blocks1(obs['obs2'])
        out_2 = self.conv_blocks2(obs['obs2'])
        out_2 = self.conv_blocks3(obs['obs2'])
        out_2 = torch.flatten(out_2)
        print(out_2.shape)

        # Merge obs 1, obs 2, and out_a
        out_4 = torch.cat((out_1, out_2, out_a))
        print(out_4.shape)
        
        #
        out_5 = self.Linear_block_2(out_4)
        out_6 = self.Last_linear_block(out_5)
        
        return out_6

#         return out_1, out_2, out_3, out_4
    
    def log(self, logger, step):
        """
        
        ----------------------------------
        
        """
        
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)