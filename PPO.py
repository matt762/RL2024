import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam
import torch.nn.functional as F
from gym import Env
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import tqdm
import time
import gym
import random

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
    
# other possible NN to add a linear layer and tanh at the end
class FeedForwardNN2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        output = self.layer4(activation3)
        output = F.tanh(output) # to test

        return output

class PPO:
    def __init__(self, env : Env):
        self._init_hyperparameters()

        # Extract env informations
        self.env = env
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0] if self.continuous else env.action_space.n

        # Init actor and critic nn
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        # Possibility 2
        #self.actor = FeedForwardNN2(self.obs_dim, self.act_dim)
        #self.critic = FeedForwardNN2(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create cov matrix for multivariate distribution and stochastic action (close to mean of actions) if continuous action space
        if self.continuous:
            self.cov_var = torch.full((self.act_dim,), fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var)

        # For logging
        self.logger = {
            'delta_t' : time.time_ns(),
            'actual_time_step' : 0,
            'actual_iteration' : 0,
            'learning_rate' : self.lr,
            'batch_lens' : [],
            'batch_rews' : [],
            'actor_losses' : [],
        }

        self.episode_rewards = []

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        actual_time_step = 0 # time_step simulated so far
        actual_iteration = 0 # iteration so far

        while actual_time_step < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()
            # Increment nb of timesteps done and iteration
            actual_time_step += np.sum(batch_lens)
            actual_iteration += 1

            self.logger['actual_time_step'] = actual_time_step
            self.logger['actual_iteration'] = actual_iteration
            self.logger['learning_rate'] = self.actor_optim.param_groups[0]["lr"]

            V, _, _ = self.evaluate(batch_obs, batch_acts)
            Adv_k = batch_rewtogo - V.detach()
            # Normalization for better stability
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)

            for _ in range(self.nb_epochs_per_iteration):
                # Compute V_phi and pi_theta(a_t | s_t)
                V, current_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

                # Compute ratios between policies
                ratios = torch.exp(current_log_probs - batch_log_probs)

                # Actor surrogate losses
                surrogate_loss1 = ratios * Adv_k
                surrogate_loss2 = torch.clamp(ratios, 1-self.clip, 1 + self.clip) * Adv_k
                
                # Losses
                actor_loss = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rewtogo) # or use F.mse_loss(V, batch_rewtogo)
                # If want to incroporate entropy in the actor loss (if coeff = 0, as if no entropy)
                entropy_loss = entropy.mean() 
                actor_loss -= self.entropy_coef * entropy_loss

                # Backward propagation using grad descent, clipping weights
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

                # Decrease lr if far in iteration
                if self.anneal_lr:
                    frac = (actual_time_step - 1.0) / total_timesteps
                    new_lr = self.lr * (1.0 - frac)
                    new_lr = max(new_lr, 0.0)
                    self.actor_optim.param_groups[0]["lr"] = new_lr
                    self.critic_optim.param_groups[0]["lr"] = new_lr
        
            
            # Information at each iteration 
            self._log_summary()

        self._update_plots()

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rewtogo = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Keep track of rewards per episode
        ep_rews = []

        timesteps_batch = 0

        while timesteps_batch < self.timesteps_per_batch:
            # Rewards collected per  episode
            ep_rews = []

            obs = self.env.reset()
            done = False
            
            for episode_t in range(self.max_timesteps_per_episode):
                # Timesteps ran in this batch so far
                timesteps_batch += 1

                # Observation in this batch
                batch_obs.append(obs)
               
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect rew, action and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and reward
            batch_lens.append(episode_t + 1)
            batch_rews.append(ep_rews)

            self.episode_rewards.append(np.sum(ep_rews))

        # Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float) if self.continuous else torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewtogo = self.compute_rewtogo(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens

    def get_action(self, obs):
        # Possibility to incorporate white or coloured noise
        # Query mean action from actor nn
        mean = self.actor(torch.tensor(obs, dtype=torch.float)) # converting to tensor just in case

        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
            action = distrib.sample()
            log_prob = distrib.log_prob(action)

            # White noise addition. If self.noise_coef, as if no noise
            if not self.coloured_noise:
                noise = np.random.normal(0, self.noise_coef)
                action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
            # Coloured noise addition
            else:
                noise = self._generate_colored_noise(size=action.detach().numpy().shape)
                action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)

            return action.detach().numpy(), log_prob.detach() # vérifier si on doit retourner avec les detach et numpy ou pas
        
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)
            action = distrib.sample().item()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.long))

            # noise addition if coef > 0 / pick a completely rdm action with probability exploration_noise
            if np.random.rand() < self.noise_coef:
                action = np.random.randint(0, self.act_dim)
                log_prob = torch.log(torch.tensor(1/self.act_dim)) # NOT SURE

            return action, log_prob.detach()
        
    def _generate_colored_noise(self, size):
        white_noise = np.random.normal(0, 1, size=size)
        frequency = np.fft.fftfreq(size[-1])
        amplitude = 1 / (np.abs(frequency) ** self.beta/2 + 1e-10)  # To avoid division by zero # use beta/2 is crucial for generating the correct type of 1/f^β noise. This scaling ensures that the noise has the desired power spectral density (PSD) characteristics
        noise_fft = np.fft.fft(white_noise)
        colored_noise_fft = noise_fft * amplitude
        colored_noise = np.fft.ifft(colored_noise_fft).real
        colored_noise *= 1e-11
        return colored_noise
        
    def evaluate(self, batch_obs, batch_acts):
        # Query value V for each obs from critic nn
        V = self.critic(batch_obs).squeeze()

        # Log prob of batch actions using actor network
        mean = self.actor(batch_obs)

        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)

        log_probs = distrib.log_prob(batch_acts)
        entropy = distrib.entropy()

        return V, log_probs, entropy

    def compute_rewtogo(self, batch_rews):
        batch_rewtogo = []

        for episode_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rewtogo.insert(0, discounted_reward)
        
        # Convert to tensor
        batch_rewtogo = torch.tensor(batch_rewtogo, dtype=torch.float)
        return batch_rewtogo

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        actual_time_step = self.logger['actual_time_step']
        actual_iteration = self.logger['actual_iteration']
        learning_rate = self.logger['learning_rate']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        learning_rate = str(round(learning_rate, 5))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{actual_iteration} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Learning rate: {learning_rate}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {actual_time_step}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def _update_plots(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, label='Total Reward per Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        length = 50

        if len(self.episode_rewards) >= length:
            means = [np.mean(self.episode_rewards[i-length:i]) for i in range(length, len(self.episode_rewards)+1)]
            stds = [np.std(self.episode_rewards[i-length:i]) for i in range(length, len(self.episode_rewards)+1)]
            x = range(length, len(self.episode_rewards)+1)
            plt.subplot(2, 1, 2)
            plt.plot(x, means, label='Mean Reward')
            plt.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
            plt.ylabel('Mean Reward')
            plt.legend()

        plt.tight_layout()
        #name = 'n' + str(self.exploration_noise) + '_ucb' + str(self.ucb_bonus_coef) + '_ent' + str(self.entropy_coef) + '_clip' + str(self.clip) + '_beta' + str(self.beta) + '_col' + str(self.coloured_noise) + '_gae' + str(self.use_gae) + '.png'
        if self.continuous:
            if self.coloured_noise:
                name = 'c' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + '_annlr' + str(self.anneal_lr) + 'col_n' + str(self.coloured_noise) + 'n_coef' + str(self.noise_coef) + 'beta' + str(self.beta) + '.png'
            else:
                name = 'c' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + '_annlr' + str(self.anneal_lr) + 'col_n' + str(self.coloured_noise) + 'n_coef' + str(self.noise_coef) + '.png'
        else:
            name = 'd' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + '_annlr' + str(self.anneal_lr) + 'n_coef' + str(self.noise_coef) + '.png'
        location = './plots_pendulum/' + name
        plt.savefig(location)


    def _init_hyperparameters(self, timesteps_per_batch = 4800, max_timesteps_per_episode = 1600, clip = 0.2,  ent_coef = 0.01, anneal_lr = False, noise_coef = 0.1, coloured_noise = False, beta = 0.5, use_gae=False):
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.use_gae = use_gae

        # Incorporate entropy, if coef = 0, as if no entropy incroporated. 0.01 is advised
        self.entropy_coef = ent_coef # A high coefficient penalized over deterministic policis --> more exploration. Useful for more complex environment
        
        # 0.2 is advised (paper)
        self.clip = clip

        # Possibility to use learning rate annealing
        self.anneal_lr = anneal_lr

        # Probably won't change
        self.gamma = 0.95
        self.nb_epochs_per_iteration = 5 # try to change it
        self.lr = 0.005 # try to change it
        self.max_grad_norm = 0.5 # try to change it # add of cliping gradient for preventing exploding gradient --> more stable learning

        # Possibility to incorporate White and Coloured noise exploration
        self.beta = beta # 0.5 is advised
        self.coloured_noise = coloured_noise
        self.noise_coef = noise_coef
        '''
        # Seed management
        self.seed = 42 # to set seed for reproductibility
        if self.seed is not None:
            assert(type(self.seed) == int)
            # set seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")'''

if __name__ == "__main__":
    env = gym.make('Pendulum-v1') # Possible env : Pendulum-v1 (continuous)/ CartPole-v1 (discrete) / MOuntainCarContinuous-v0 (continuous) / MountainCar-v0 (discrete)
    model = PPO(env)
    model._init_hyperparameters(coloured_noise=True, beta=0.5)
    model.learn(100000)

'''
Possible tests :
for clip in [x, y]
    for entropy in [x, y]
        for anneal_lr in [True, False]
            # for noise
            for noise_coef in [x, y]
                for coloured_noise in [False, True]
                    if coloured_noise
                        for beta in [x, y]
'''