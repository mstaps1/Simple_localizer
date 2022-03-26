import numpy as np
import random

import gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim

import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import *

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, NN_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=20):
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
        
        self.device = "cpu"

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # with my env I have two inputs
        
        sequence_length = NN_size['sequence_length']

        self.sequence_length = sequence_length

        state_size_1 = state_size['state_1']
        hidden_size_1 = NN_size['hidden_1_size']
        num_layers_1 = NN_size['hidden_1_num_layers']

        self.rnn_input_1 = nn.LSTM(input_size = state_size_1, hidden_size = hidden_size_1, num_layers = num_layers_1, batch_first=True)



        # Convolution parameters
        input_channels = NN_size['input_channels']
        hidden_channels = NN_size['hidden_channels']
        output_channels = NN_size['output_channels']

        kernel_size = NN_size['kernel_size']
        stride = NN_size['stride']

        # 
        self.conv_disc = nn.Sequential(
            self.make_conv_block(sequence_length, sequence_length, kernel_size = kernel_size, stride = stride),
            self.make_conv_block(sequence_length, sequence_length, kernel_size = kernel_size, stride = stride),
            self.make_conv_block(sequence_length, sequence_length, kernel_size = kernel_size, stride = stride)
        )
        
        W = state_size['state_2']
        K = kernel_size
        P = 0
        S = stride
        size_out_1 = (W - K + 2*P)/S+1
        size_out_2 = (size_out_1 - K + 2*P)/S+1
        size_out_3 = (size_out_2 - K + 2*P)/S+1

        input_size_conv_out = int(size_out_3**2)
        print('input_size_conv_out ',input_size_conv_out)
        hidden_size_2 = NN_size['hidden_2_size']
        num_layers_2 = NN_size['hidden_2_num_layers']
        
        
        input_size_rnn_2 = 576
        # what is the input size based on?
        self.rnn_input_2 = nn.LSTM(input_size = input_size_rnn_2,
                                   hidden_size = 128,
                                   num_layers= 4,
                                   batch_first=True)       

        input_dims_last = NN_size['input_dims_last']
        hidden_3_size = NN_size['hidden_3_size']
        hidden_3_out = NN_size['hidden_3_out']
        hidden_3_num_layers = NN_size['hidden_3_num_layers']

        #
        self.fc_block_last = MLP(input_dims_last,
                             hidden_3_size,
                             hidden_3_out,
                             hidden_3_num_layers)

        
        # Output Layer
        self.mu = nn.Linear(hidden_3_out, action_size)
        # 
        self.log_std_linear = nn.Linear(hidden_3_out, action_size)


    def make_conv_block(self, input_channels, output_channels, kernel_size=4, stride=2):
        return  nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )

    def forward(self, state_1, state_2):
        """

        """

        Batch_size = state_2.shape[0]


        # State 1 input
        x1, (hn_a, cn_a) = self.rnn_input_1(state_1)


        # 
        x_1 = hn_a[-1]

        # convolution input (state 2)       
        x2 = self.conv_disc(state_2)
        
        # reshape the output of the convolution
        x_2 = x2.reshape(Batch_size, self.sequence_length, -1)
        
        #

        output, (hn_b, cn_b) = self.rnn_input_2(x_2)
        
        #
        output_r = hn_b[-1]
        
        #
        x3 = torch.cat((x_1, output_r), dim=1)
        
        #
        x4 = self.fc_block_last(x3)

        mu = self.mu(x4)
        
        log_std = self.log_std_linear(x4)        
        
        self.log_std_min = -20
        self.log_std_max = 20
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std

    
    def evaluate(self, state_1, state_2, epsilon=1e-6):
        """
        
        """
        mu, log_std = self.forward(state_1, state_2)
        
        std = log_std.exp()
        
        dist = Normal(0, 1)
        
        e = dist.sample().to(self.device)
        
        action = torch.tanh(mu + e * std)
        
        
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        # needed so log prob can be used with critic
        log_prob = log_prob.sum(1,keepdim=True)
        return action, log_prob
        
    
    def get_action(self, state_1, state_2):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        #state = torch.FloatTensor(state).to(device) #.unsqzeeze(0)
        
        mu, log_std = self.forward(state_1, state_2)
               
        std    = log_std.exp()
        dist   = Normal(0, 1)
        e      = dist.sample().to(self.device)
        
        action = torch.tanh(mu + e * std).cpu()
  
        #action = torch.clamp(action*action_high, action_low, action_high)
        return action

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, NN_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.device = "cpu"
        
        # with my env I have two inputs
        state_size_1 = state_size['state_1']
        hidden_size_1 = NN_size['hidden_1_size']
        num_layers_1 = NN_size['hidden_1_num_layers']
        sequence_length = NN_size["sequence_length"]

        self.sequence_length = sequence_length

        self.rnn_input_1 = nn.LSTM(input_size = state_size_1, hidden_size = hidden_size_1, num_layers = num_layers_1, batch_first=True)



        # Convolution parameters
        input_channels = NN_size['input_channels']
        hidden_channels = NN_size['hidden_channels']
        output_channels = NN_size['output_channels']

        kernel_size = NN_size['kernel_size']
        stride = NN_size['stride']

        # 
        self.conv_disc = nn.Sequential(
            self.make_conv_block(sequence_length, sequence_length, kernel_size=kernel_size, stride=stride),
            self.make_conv_block(sequence_length, sequence_length, kernel_size=kernel_size, stride=stride),
            self.make_conv_block(sequence_length, sequence_length, kernel_size=kernel_size, stride=stride)
        )

        W = state_size['state_2']
        K = kernel_size
        P = 0
        S = stride
        size_out_1 = (W - K + 2*P)/S+1
        size_out_2 = (size_out_1 - K + 2*P)/S+1
        size_out_3 = (size_out_2 - K + 2*P)/S+1

        input_size_conv_out = int(size_out_3**2)
        hidden_size_2 = NN_size['hidden_2_size']
        num_layers_2 = NN_size['hidden_2_num_layers']

        # 
        self.rnn_input_2 = nn.LSTM(input_size = 576, hidden_size = hidden_size_2, num_layers= num_layers_2, batch_first=True)       

        input_dims_last = NN_size['input_dims_last critic']
        hidden_3_size = NN_size['hidden_3_size']
        hidden_3_out = NN_size['hidden_3_out']
        hidden_3_num_layers = NN_size['hidden_3_num_layers']

        #
        self.fc_block_last = MLP(input_dims_last,
                             hidden_3_size,
                             hidden_3_out,
                             hidden_3_num_layers)
        #************************************************************************************************

        self.fc_last = nn.Linear(hidden_3_out, 1)


        
    def make_conv_block(self, input_channels, output_channels, kernel_size=4, stride=2):
        return  nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )

    def forward(self, state1, state2, action):
        """
        Build a critic (value) network that maps (state, action) pairs -> Q-values.
        """
        # concatenate state vector with action vector        
        # x_in = torch.cat((state1, action), dim=2)
        
        Batch_size = state1.shape[0]
        
        # State 1 input
        x1, (hn_a, cn_a) = self.rnn_input_1(state1)
        x_1 = hn_a[-1]

        # convolution input (state 2)
        x2 = self.conv_disc(state2)

        x_2 = x2.reshape(Batch_size, self.sequence_length, -1)
        
        output, (hn_b, cn_b) = self.rnn_input_2(x_2)
        output_r = hn_b[-1]

        x3 = torch.cat((x_1, output_r, action), dim=1)
        
        x4 = self.fc_block_last(x3)


        x_5 = self.fc_last(x4)
        return x_5
    
    
    
class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, NN_size, action_prior="uniform"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.LR_ACTOR = 5e-4
        self.LR_CRITIC = 5e-4
        
        self.BUFFER_SIZE = int(1e6)
        self.BATCH_SIZE = 256
        self.GAMMA = 0.99
        self.TAU = 1e-2
        self.FIXED_ALPHA = None
        
        self.state_size = state_size #dict()
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.LR_ACTOR) 
        self._action_prior = action_prior
        
        device = 'cpu'
        self.device = 'cpu'
        
        action_low = np.array([-2,0])
        action_high = np.array([2,np.pi/2])
 
        log_std_min = np.array([])
        log_std_max = np.array([])
        
        self.sequence_length = NN_size['sequence_length']
        
        self.state_size = state_size
        
        # (state_size, action_size, seed, NN_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=20)
        
        #*****************************************************
        # Actor Network
        #*****************************************************
        self.actor_local = Actor(state_size,
                                 action_size,
                                 random_seed,
                                 NN_size = NN_size,
                                 log_std_min = log_std_min,
                                 log_std_max = log_std_max).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.LR_ACTOR)     
        
        #*****************************************************
        # Critic Network (w/ Target Network)
        #*****************************************************
        self.critic1 = Critic(state_size, action_size, random_seed, NN_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, NN_size).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed, NN_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed, NN_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(),
                                            lr=self.LR_CRITIC,
                                            weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(),
                                            lr=self.LR_CRITIC,
                                            weight_decay=0) 
        
       
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed, self.sequence_length, state_size)
        

    def step(self, state_1, state_2, action, reward, next_state_1, next_state_2, done, step):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience / reward
        self.memory.add(state_1, state_2, action, reward, next_state_1, next_state_2, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            #
            experiences = self.memory.sample()
            #
            self.learn(step, experiences, self.GAMMA)
            
    
    def act(self, state_1, state_2):
        """
        Returns actions for given state as per current policy.
        """
        state_1 = torch.from_numpy(state_1).float().to(self.device)
        state_2 = torch.from_numpy(state_2).float().to(self.device)
        
        # 
        state_2 = state_2.reshape(1, self.sequence_length, self.state_size['state_2'], self.state_size['state_2'])        
        
        action = self.actor_local.get_action(state_1, state_2).detach()
        
        return action

    def learn(self, step, experiences, gamma, d=1):
        """
        Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_1, states_2, actions, rewards, next_state_1, next_state_2, dones = experiences
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        #       
        
        next_action, log_pis_next = self.actor_local.evaluate(next_state_1, next_state_2)

        # 
#         Q_target1_next = self.critic1_target(next_state_1.to(self.device),next_state_2.to(self.device), next_action.squeeze(0).to(self.device))
#         Q_target2_next = self.critic2_target(next_state_1.to(self.device),next_state_2.to(self.device), next_action.squeeze(0).to(self.device))
        
        # Removed Sqeeze on next action

        Q_target1_next = self.critic1_target(next_state_1.to(self.device),
                                             next_state_2.to(self.device),
                                             next_action.to(self.device))
        #
        
        Q_target2_next = self.critic2_target(next_state_1.to(self.device),
                                             next_state_2.to(self.device),
                                             next_action.to(self.device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next,
                                  Q_target2_next)
        
        if self.FIXED_ALPHA == None:
            
            
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
            
        else:
            
            #
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
        
        
        # Compute critic loss
        Q_1 = self.critic1(states_1, states_2, actions).cpu()
        Q_2 = self.critic2(states_1, states_2, actions).cpu()
        

        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if self.FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states_1, states_2)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states_1, states_2, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:
                
                actions_pred, log_pis = self.actor_local.evaluate(states_1, states_2)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
    
                actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states_1, states_2, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, self.TAU)
            self.soft_update(self.critic2, self.critic2_target, self.TAU)
                     

    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, sequence_length, state_size):
        """
        Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_1", "state_2", "action", "reward", "next_state_1", "next_state_2", "done"])

        self.seed = random.seed(seed)
        self.device = 'cpu'
        self.sequence_length = sequence_length
        self.state_size = state_size
    def add(self, state_1, state_2, action, reward, next_state_1, next_state_2, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state_1, state_2, action, reward, next_state_1, next_state_2, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """

        #
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Store state_1
        states_1 = torch.from_numpy(np.vstack([e.state_1 for e in experiences if e is not None])).float().to(self.device)

        # Store state_2
        states_2 = torch.from_numpy(np.vstack([e.state_2.reshape(-1,self.sequence_length,
                                                                 self.state_size['state_2'],
                                                                 self.state_size['state_2']) for e in experiences if e is not None])).float().to(self.device)
        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)

        next_states_1 = torch.from_numpy(np.vstack([e.next_state_1 for e in experiences if e is not None])).float().to(self.device)
        
        # 
        
        
        next_states_2 = torch.from_numpy(np.vstack([e.next_state_2.reshape(1,
                                                                           self.sequence_length, 
                                                                           self.state_size['state_2'], 
                                                                           self.state_size['state_2']) for e in experiences if e is not None])).float().to(self.device)
        
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states_1, states_2, actions, rewards, next_states_1, next_states_2, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
    
    
    
def SAC(n_episodes=200, max_t=500, print_every=10):
    #
    scores_deque = deque(maxlen=100)
    average_100_scores = []

    for i_episode in range(1, n_episodes+1):
        #
        state = env.reset()
        
        state_1 = state['obs1']
        state_2 = state['obs2']
        
        state_1 = state_1.reshape((-1,state_size['state_1']))
        state_2 = state_2.reshape((-1,state_size['state_2']))
        
        score = 0
        
        for t in range(max_t):
            #
            action = agent.act(state_1, state_2)
            action_v = action.numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            
            next_state_1 = next_state['obs1']
            next_state_2 = next_state['obs2']

            next_state_1 = next_state_1.reshape((-1, state_size['state_1']))
            next_state_2 = next_state_2.reshape((-1, state_size['state_2']))
            
            agent.step(state_1, state_2, action, reward, next_state_1, next_state_2, done, t)
            
            state_1, state_2 = next_state_1, next_state_2
            
            state = next_state
            
            score += reward

            if done:
                break 
        
        scores_deque.append(score)
        writer.add_scalar("Reward", score, i_episode)
        writer.add_scalar("average_X", np.mean(scores_deque), i_episode)
        average_100_scores.append(np.mean(scores_deque))
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
            
            
    torch.save(agent.actor_local.state_dict(), args.info + ".pt")
    



def play():
    agent.actor_local.eval()
    for i_episode in range(1):

        state = env.reset()
        
        state_1 = state['obs1']
        state_2 = state['obs2']
        
        state_1 = state_1.reshape((1,state_size['state_1']))
        state_2 = state_2.reshape((1,state_size['state_2']))

        while True:
            action = agent.act(state_1, state_2)
            action_v = action[0].numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            
            next_state_1 = next_state['obs1']
            next_state_2 = next_state['obs2']
            
            next_state_1 = next_state_1.reshape((-1, state_size['state_1']))
            next_state_2 = next_state_2.reshape((-1, state_size['state_2']))
            
            state_1 = next_state_1
            state_2 = next_state_2
            if done:
                break 