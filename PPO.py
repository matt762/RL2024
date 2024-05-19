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

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
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
    def __init__(self, env):
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
        
        while actual_time_step < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()
            actual_time_step += np.sum(batch_lens)
            actual_iteration += 1

            self.logger['actual_time_step'] = actual_time_step
            self.logger['actual_iteration'] = actual_iteration

            V, _, _ = self.evaluate(batch_obs, batch_acts)
            Adv_k = batch_rewtogo - V.detach()
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)

            for _ in range(self.nb_epochs_by_iteration):
                V, current_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(current_log_probs - batch_log_probs)
                surrogate_loss1 = ratios * Adv_k
                surrogate_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * Adv_k
                
                critic_loss = nn.MSELoss()(V, batch_rewtogo)
                actor_loss = -(torch.min(surrogate_loss1, surrogate_loss2) + self.entropy_coef * entropy).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
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

        time_step_batch = 0
        ep_rews = []

        while time_step_batch < self.timesteps_per_batch:
            ep_rews = []           
            obs = self.env.reset()
            done = False

            for episode_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['actual_iteration'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render(mode='human')
                    
                time_step_batch += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                self.state_visit_count[tuple(obs)] += 1

                if done:
                    break

            batch_lens.append(episode_t + 1)
            batch_rews.append(ep_rews)
            self.episode_rewards.append(np.sum(ep_rews))

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float) if self.continuous else torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewtogo = self.compute_rewtogo(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens

    def get_action(self, obs):
        mean = self.actor(torch.tensor(obs, dtype=torch.float))
        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
            action = distrib.sample().detach().numpy()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.float))
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
            return action, log_prob.detach()
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)
            action = distrib.sample().item()
            log_prob = distrib.log_prob(torch.tensor(action, dtype=torch.long))
            if np.random.rand() < self.exploration_noise:
                action = np.random.randint(0, self.act_dim)
                log_prob = torch.log(torch.tensor(1 / self.act_dim))
            return action, log_prob.detach()

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
        state_visit_count_tensor = torch.tensor([self.state_visit_count[tuple(obs.numpy())] for obs in batch_obs])
        ucb_bonus = self.ucb_bonus_coef / torch.sqrt(state_visit_count_tensor + 1)

        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
            log_probs = distrib.log_prob(batch_acts)
            entropy = distrib.entropy().mean()
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)
            log_probs = distrib.log_prob(batch_acts)
            entropy = distrib.entropy().mean()

        return V + ucb_bonus, log_probs, entropy

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
        name = str(self.exploration_noise) + '_' + str(self.ucb_bonus_coef) + '_' + str(self.entropy_coef) + '_' + str(self.clip) + '.png'
        location = './plot_cartpole/' + name
        plt.savefig(location)
        
    def _init_hyperparameters(self, noise = 0.1, ucb = 0.1, entropy=0.01, clip=0.2, render=False):
        self.timesteps_per_batch = 4600
        self.max_timesteps_per_episode = 2000
        self.gamma = 0.99
        self.nb_epochs_by_iteration = 10
        self.clip = clip
        self.lr = 0.005
        self.save_frequency = 100
        self.render = render
        self.render_every_i = 108
        self.exploration_noise = noise
        self.ucb_bonus_coef = ucb
        self.entropy_coef = entropy

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = PPO(env)
    
    noise = 0
    ucb = 0
    entropy = 0
    clip = 0.2
    
    for entropy in [0, 0.05, 0.1, 0.2, 0.3]:
        print(f"Exploration noise: {noise}, UCB bonus coef: {ucb}, Entropy coef: {entropy}, Clip: {clip}")
        model._init_hyperparameters(noise, ucb, entropy, clip, render=False)
        model.learn(500000)
