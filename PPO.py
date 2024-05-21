import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam
import time
import gym
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import torch.nn.functional as F
import tqdm
from gym import Env
import random

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)
        # self.relu = F.relu()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        #elif isinstance(obs, tuple):
            # Concatenate the elements of the tuple along the first dimension to create a tensor
        #    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)


        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

class PPO:
    def __init__(self, env : Env):
        self.env = env
        self._init_hyperparameters()
        
        self.obs_dim = env.observation_space.shape[0]
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        self.act_dim = env.action_space.shape[0] if self.continuous else env.action_space.n
        
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.logger = {
            'delta_t': time.time_ns(),
            'actual_time_step': 0,
            'actual_iteration': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
        }
        self.episode_rewards = []
        self.state_visit_count = defaultdict(int)
        
        if self.continuous:
            self.cov_var = torch.full((self.act_dim,), 0.5)
            self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_time_steps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, {self.timesteps_per_batch} timesteps per batch for a total of {total_time_steps} timesteps")
        actual_time_step = 0
        actual_iteration = 0

        # self.set_seed(42) # set constante seed for now
        
        while actual_time_step < total_time_steps:
            if not self.use_gae:
                batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()
            else:
                batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
                _,_,_,ucb = self.evaluate(batch_obs, batch_acts)
                Adv_k = self.compute_gae(batch_rews, batch_vals, batch_dones)
                # print(f"UCB : {ucb}")
                # print(f"Adv_k : {Adv_k}")
                V = self.critic(batch_obs).squeeze()
                batch_rewtogo = Adv_k + V.detach() # VOIR OU IL FAIT CA DEMAINLA SUITE
            
            actual_time_step += np.sum(batch_lens)
            actual_iteration += 1

            self.logger['actual_time_step'] = actual_time_step
            self.logger['actual_iteration'] = actual_iteration

            if not self.use_gae:
                V, _, _, ucb = self.evaluate(batch_obs, batch_acts)
                Adv_k = batch_rewtogo - V.detach()
                Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)
                # self.print_full_tensor(ucb, "UCB :")
                # self.print_full_tensor(Adv_k, "Advantage :")

            for _ in range(self.nb_epochs_by_iteration):
                V, current_log_probs, entropy, _ = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(current_log_probs - batch_log_probs)  # TO CHECK
                surrogate_loss1 = ratios * Adv_k
                surrogate_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * Adv_k
                
                #critic_loss = nn.MSELoss()(V, batch_rewtogo)
                critic_loss = F.mse_loss(V, batch_rewtogo) # to test if diff
                actor_loss = -(torch.min(surrogate_loss1, surrogate_loss2) + self.entropy_coef * entropy).mean() # METHOD 1 ? MATTEO
                # actor_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean() - self.entropy_coef * entropy.mean() # METHOD 2 JEAN ?

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

            self._log_summary()

            if actual_iteration % self.save_frequency == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
            
        self._update_plots()

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rewtogo = []
        batch_lens = []

        if self.use_gae:
            batch_vals = []
            batch_dones = []
            ep_vals = []
            ep_dones = []

        time_step_batch = 0
        ep_rews = []

        while time_step_batch < self.timesteps_per_batch:
            if self.use_gae:
                ep_vals = []
                ep_dones = []

            ep_rews = []           
            obs = self.env.reset()
            done = False

            for episode_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['actual_iteration'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render(mode='human')

                if self.use_gae:
                    ep_dones.append(done)
                    val = self.critic(obs)
                    
                time_step_batch += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)

                if not self.use_gae:
                    obs, rew, done, _ = self.env.step(action)
                else:
                    obs, rew, terminated, truncated = self.env.step(action)
                    done = terminated or bool(truncated)
                    ep_vals.append(val.flatten())

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                # print(sum(1 for count in self.state_visit_count.values() if count > 11.0))
                
                # Update visitation count
                obs_key = tuple(int(np.round(o, decimals=2)*100) for o in obs)
                self.state_visit_count[obs_key] += 1
                
                # self.print_full_tensor(self.state_visit_count, "State visit count :")

                if done:
                    break

            batch_lens.append(episode_t + 1)
            batch_rews.append(ep_rews)
            if self.use_gae:
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)


            self.episode_rewards.append(np.sum(ep_rews))
            
            # print(f"Seed: {self.env.seed()}")

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float) if self.continuous else torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewtogo = self.compute_rewtogo(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        if not self.use_gae:
            return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens # ATTENDS Là ON RENVOIT BATCH_REWTOGO MAIS PAS BATCH_REWS ????? PARCE QU'ON UTILISE REWS QUE DANS REWTOGO ???
        else:
            return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs):
        mean = self.actor(torch.tensor(obs, dtype=torch.float))
        '''# ancient version with white noise
        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
            action = distrib.sample().detach().numpy()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.float))

            # noise addition (at every action for the moment)
            noise = np.random.normal(0, self.exploration_noise, size=action.shape) # Gaussian noise, parameters : mean / std dev / output shape
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high) # cliping to prevent getting out of range of action_space with addition of noise.

            return action, log_prob.detach()'''
        


        '''# new version with colored noise. Noise on the mean then we use distribution
        
        if self.continuous:
            noise = self.generate_colored_noise(size=mean.shape)
            #noise *= self.exploration_noise # DONT KNOW IF WE HAVE TO SCALE NOISE
            noisy_mean = noise + mean

            distrib = MultivariateNormal(noisy_mean, self.cov_mat)
            action = distrib.sample().detach.numpy()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.float))
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)     
            
            return action, log_prob.detach()'''
        # new version with colored noise. Noise added after choice of action
        if self.continuous and self.coloured_noise:
            distrib = MultivariateNormal(mean, self.cov_mat)
            action = distrib.sample().detach().numpy()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.float)) 

            # Generated correlated noise and add it to action
            noise = self.generate_colored_noise(size=action.shape)
            #noise *= self.exploration_noise # # DONT KNOW IF WE HAVE TO SCALE NOISE
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
            return action, log_prob # .detach ou pas ?
        
        # old version without coloured noise
        elif self.continuous and not self.coloured_noise:
            distrib = MultivariateNormal(mean, self.cov_mat)
            action = distrib.sample().detach().numpy()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.float))

            # noise addition (at every action for the moment)
            noise = np.random.normal(0, self.exploration_noise, size=action.shape) # Gaussian noise, parameters : mean / std dev / output shape
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high) # cliping to prevent getting out of range of action_space with addition of noise.

            return action, log_prob.detach()
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)
            action = distrib.sample().item()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.long))

            # noise addition
            if np.random.rand() < self.exploration_noise: # pick a completely random action with probability self.exploration_noise
                action = np.random.randint(0, self.act_dim) # TO CHECK
                log_prob = torch.log(torch.tensor(1 / self.act_dim))

            return action, log_prob.detach()
        
    def generate_colored_noise(self, size):
        white_noise = np.random.normal(0, 1, size=size)
        frequency = np.fft.fftfreq(size[-1])
        amplitude = 1 / (np.abs(frequency) ** self.beta/2 + 1e-10)  # To avoid division by zero # use beta/2 is crucial for generating the correct type of 1/f^β noise. This scaling ensures that the noise has the desired power spectral density (PSD) characteristics
        noise_fft = np.fft.fft(white_noise)
        colored_noise_fft = noise_fft * amplitude
        colored_noise = np.fft.ifft(colored_noise_fft).real
        return colored_noise

# try to use gae
    def compute_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - int(ep_dones[t+1])) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lambda_gae * (1 - int(ep_dones[t])) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def compute_rewtogo(self, batch_rews):
        batch_rewtogo = []

        for episode_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(episode_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rewtogo.insert(0, discounted_reward)

        batch_rewtogo = torch.tensor(batch_rewtogo, dtype=torch.float)
        return batch_rewtogo

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)        
        state_visit_count_tensor = torch.tensor([self.state_visit_count[tuple(int(np.round(o, decimals=2)*100) for o in obs)] for obs in batch_obs])
        ucb_bonus = self.ucb_bonus_coef / torch.sqrt(state_visit_count_tensor + 1)
        
        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)

        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)

        log_probs = distrib.log_prob(batch_acts)
        entropy = distrib.entropy() #.mean() je crois faut enlever le mean() là parce qu'on le fait dans le learn (matteo l'a mis je l'enlève)
        
        # print(f"Entropy : {entropy*self.entropy_coef}")

        return V, log_probs, entropy, ucb_bonus
    
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set the seed for the gym environment if applicable
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        if hasattr(self.env.action_space, 'seed'):
            self.env.action_space.seed(seed)
        if hasattr(self.env.observation_space, 'seed'):
            self.env.observation_space.seed(seed)

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        actual_time_step = self.logger['actual_time_step']
        actual_iteration = self.logger['actual_iteration']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{actual_iteration} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
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
        name = 'n' + str(self.exploration_noise) + '_ucb' + str(self.ucb_bonus_coef) + '_ent' + str(self.entropy_coef) + '_clip' + str(self.clip) + '_beta' + str(self.beta) + '_col' + str(self.coloured_noise) + '_gae' + str(self.use_gae) + '.png'
        location = './plots_pendulum/' + name
        plt.savefig(location)
        
    def print_full_tensor(self, x, name):
        print(name)
        torch.set_printoptions(profile="full")
        print(x)
        torch.set_printoptions(profile="default") #reset
        
    def _init_hyperparameters(self, noise = 0.1, ucb = 0.1, entropy=0.01, clip=0.2, render=False, lr = 0.005, timesteps_per_batch = 4600, max_timesteps_per_episode=2000, beta = 0, coloured_noise=True, use_gae=True, critic = 0.5):
        # Can be modified for training
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.clip = clip
        self.lr = lr
        self.exploration_noise = noise
        self.ucb_bonus_coef = ucb
        self.entropy_coef = entropy
        self.critic_weight = critic

        self.gamma = 0.99
        self.nb_epochs_by_iteration = 1 # number of seeds to test on
        self.save_frequency = 100
        self.render = render
        self.render_every_i = 30
        
        self.beta = beta
        self.coloured_noise = coloured_noise
        self.use_gae = use_gae
        self.lambda_gae = 0.95
        self.max_grad_norm = 0.5 # add of cliping gradient for preventing exploding gradient --> more stable learning
        # GAE lambda 0.95, entropy 0.01, clip 0.2 (or 0.1 * alpha), lr 0.00025 * alpha (alpha annealing from 1 to 0 over the course of learning) for atari games as suggested by paper

'''
if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    model = PPO(env)
    
    for noise in [0, 0.05, 0.1, 0.2, 0.3]:
        for ucb in [0, 0.05, 0.1, 0.2, 0.3]:
            for entropy in [0, 0.005, 0.01, 0.02, 0.03]:
                for clip in [0.1, 0.2, 0.3]:
                    print(f"Exploration noise: {noise}, UCB bonus coef: {ucb}, Entropy coef: {entropy}, Clip: {clip}")
                    model._init_hyperparameters(noise, ucb, entropy, clip, render=False)
                    model.learn(500000)
'''

if __name__ == "__main__":
    # env = gym.make('CartPole-v0') # Possible env : Pendulum-v1 (continuous)/ CartPole-v1 (discrete) / MOuntainCarContinuous-v0 (continuous) / MountainCar-v0 (discrete)
    env = gym.make('Pendulum-v1') # Possible env : Pendulum-v1 (continuous)/ CartPole-v1 (discrete) / MOuntainCarContinuous-v0 (continuous) / MountainCar-v0 (discrete)
    model = PPO(env)
    
    # for noise in [1e-11, 2e-11]:
    #     for ucb in [0.05, 0.1]:
    #         for entropy in [0.005, 0.01]:
    #             for clip in [0.2]:
    #                 for beta in [0, 0.2, 0.5, 1]:
    #                     for coloured_noise in [True]:
    #                         for use_gae in [True, False]:
    #                             print(f"Exploration noise: {noise}, UCB bonus coef: {ucb}, Entropy coef: {entropy}, Clip: {clip}, Beta_coloured_noise: {beta}, Coloured noise or not : {coloured_noise}, use_gae: {use_gae}")
    #                             model._init_hyperparameters(noise, ucb, entropy, clip, render=False, lr=0.00025, beta=beta, coloured_noise=coloured_noise, use_gae=use_gae) # use lr = 0.00025 or 0.0005
    #                             model.learn(200000)
    
noise = 1e-11
ucb = 1
entropy = 0.01
clip = 0.2
beta = 1
coloured_noise = True
use_gae = False
critic = 1e-5

render = True
    
print(f"Exploration noise: {noise}, UCB bonus coef: {ucb}, Entropy coef: {entropy}, Clip: {clip}, Beta_coloured_noise: {beta}, Coloured noise or not : {coloured_noise}, use_gae: {use_gae}")
model._init_hyperparameters(noise, ucb, entropy, clip, render=False, lr=0.00025, beta=beta, coloured_noise=coloured_noise, use_gae=use_gae, critic = critic) # use lr = 0.00025 or 0.0005
model.learn(200000)