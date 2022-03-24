import os
import numpy as np
from datetime import datetime


import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils import *
from networks_HER import ActorNetwork, CriticNetwork

class SAC_HER(nn.Module):
    def __init__(self, args, env, env_params):
        super(SAC_HER, self).__init__()
        """
        SAC with HER added
        ----------------------------------------------
        env_params: dict()
                            action_max
                            obs1
                            obs2
                            goal
                            action_dim
        args: dict()
                            discount
                            device
                            critic_tau
                            actor_update_frequency
                            critic_target_update_frequency
                            batch_size
                            learnable_temperature
                            init_temperature
                            lr_actor
                            lr_critic
                            alpha_lr
                            alpha_betas
                            
        """
        self.args = args
        self.env = env
        self.env_params = env_params
        
        self.device = torch.device(args['device'])
        
        self.critic_tau = args['critic_tau']
        self.actor_update_frequency = args['actor_update_frequency']
        self.critic_target_update_frequency = args['critic_target_update_frequency']
        self.batch_size = args['batch_size']
        self.learnable_temperature = args['learnable_temperature']
        self.discount = args['discount']
        
        
        # env_params['action_max'], env_params['obs1'], env_params['obs2'], env_params['goal']
        
        #Create and initialize the networks
        self.ActorNetwork = ActorNetwork(args['critic_betas'], env_params, name='critic', chkpt_dir='tmp/sac')
        self.CriticNetwork_q1 = CriticNetwork(args['actor_betas'], env_params, name='critic', chkpt_dir='tmp/sac')
        self.CriticNetwork_q2 = CriticNetwork(args['actor_betas'], env_params, name='critic', chkpt_dir='tmp/sac')
        self.critic_target_q1 = CriticNetwork(args['actor_betas'], env_params, name='critic', chkpt_dir='tmp/sac')
        self.critic_target_q2 = CriticNetwork(args['actor_betas'], env_params, name='critic', chkpt_dir='tmp/sac')
        self.critic_target_q1.load_state_dict(self.CriticNetwork_q1.state_dict())
        self.critic_target_q2.load_state_dict(self.CriticNetwork_q2.state_dict())
        # Not sure what log alpha does
        self.log_alpha = torch.tensor(np.log(args['init_temperature'])).to(self.device)
        self.log_alpha.requires_grad = True
        action_dim = env_params['action_dim']
        
        # set target entropy to |A|
        self.target_entropy = -action_dim

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.ActorNetwork.parameters(), lr=self.args['lr_actor'])
        self.critic_optim_q1 = torch.optim.Adam(self.CriticNetwork_q1.parameters(), lr=self.args['lr_critic'])
        self.critic_optim_q2 = torch.optim.Adam(self.CriticNetwork_q2.parameters(), lr=self.args['lr_critic'])
        
        
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                        lr=args['alpha_lr'],
                                                        betas=args['alpha_betas'])
        self.discount = args['discount']
        
        # Train
        self.train()
        self.critic_target_q1.train()
        self.critic_target_q2.train()
        
    def train(self, training=True):
        """
        Train stuff
        """
        self.training = training
        self.ActorNetwork.train(training)
        self.CriticNetwork_q1.train(training)
        self.CriticNetwork_q2.train(training)
    
    @property
    def alpha(self):
        """
        return the exponent of log alpha
        """
        return self.log_alpha.exp()
    
    def act(self, obs, sample=False):
        """
        Update the Actor
        obs: dict() -> 'obs1', 'obs2', and 'goal'
        """
        x = dict()
        x['obs1'] = torch.tensor(obs['obs1'], dtype=torch.float32)
        x['obs2'] = torch.tensor(obs['obs2'], dtype=torch.float32)
        x['goal'] = torch.tensor(obs['goal'], dtype=torch.float32)
        
        # Testing using -1
        x['obs2'] = x['obs2'].reshape(-1,1,50,50)
        x['goal'] = x['goal'].reshape(-1,1,50,50)
        
        dist = self.ActorNetwork.forward(x)
        print("dist: ", dist)
        action = dist.sample() if sample else dist.mean
        print('action ',action)
        action = torch.clamp(action, self.env_params['action_max'])
        #action = action.clamp(*self.env_params['action_max'])
        
        # make sure action shape is correct
        assert action.ndim == 2 and action.shape[0] == 2
        return utils.to_np(action[0])
    
    def update_critic(self, obs, action, reward, next_obs, logger, step):
        """
        Update the Critic
        obs: dict() -> 'obs1', 'obs2', and 'goal'
        """
        #Create dict to send and make sure inputs are torch. 
        x_next = dict()
        x_next['obs1'] = torch.tensor(next_obs['obs1'], dtype=torch.float32)
        x_next['obs2'] = torch.tensor(next_obs['obs2'], dtype=torch.float32)
        x_next['goal'] = torch.tensor(next_obs['goal'], dtype=torch.float32)
        
        x_next['obs1'] = x_next['obs1'].reshape(-1,5)
        x_next['obs2'] = x_next['obs2'].reshape(-1,1,50,50)
        x_next['goal'] = x_next['goal'].reshape(-1,1,50,50)
        
        x = dict()
        x['obs1'] = torch.tensor(obs['obs1'], dtype=torch.float32)
        x['obs2'] = torch.tensor(obs['obs2'], dtype=torch.float32)
        x['goal'] = torch.tensor(obs['goal'], dtype=torch.float32)
        
        x['obs1'] = x['obs1'].reshape(-1,5)
        x['obs2'] = x['obs2'].reshape(-1,1,50,50)
        x['goal'] = x['goal'].reshape(-1,1,50,50)
        
        # get action distribution based on next obs
        dist = self.ActorNetwork.forward(x)
        # sample next action based on distribution
        next_action = dist.rsample()
        
        #
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        
        
        # Compute targets for the Q function
        target_Q1 = self.CriticNetwork.forward(x_next, next_action)
        target_Q2 = self.critic_target.forward(x_next, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.CriticNetwork.forward(x, action), self.CriticNetwork.forward(x, action)
        
        #Calculate critic
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        #
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #log critic stuff
        self.critic.log(logger, step)
    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        transitions = replay_buffer.sample(self.batch_size)
        
        obs1, obs2, obs1_next, obs2_next = transitions['obs1'], transitions['obs1'], transitions['obs1_next'], transitions['obs2_next']
        g = transitions['g']
        
        
        logger.log('train/batch_reward', reward.mean(), step)
        
        #Send to the update Critic function
        self.update_critic(obs, action, reward, next_obs, logger, step)
        
        #
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)
        
        #
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.CriticNetwork_q1, self.critic_target_q1, self.critic_tau)
            utils.soft_update_params(self.CriticNetwork_q1, self.critic_target_q2, self.critic_tau)