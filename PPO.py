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
import time
import gym
import random
import seaborn as sns
import pandas as pd
from scipy.signal import lfilter

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
        self.env = env
        self._init_hyperparameters()

        # Extract env informations
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
        
        self.actual_time_step = 0

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
        self.time_step_episode = []
        
        self.state_visit_count = defaultdict(int)

        self.noise_array = self.generate_colored_noise()
        print(len(self.noise_array))

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        self.actual_time_step = 0 # time_step simulated so far
        actual_iteration = 0 # iteration so far

        while self.actual_time_step < total_timesteps:
            
            if not self.use_gae:
                batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()
                V, _, _, ucb = self.evaluate(batch_obs, batch_acts)
                Adv_k = batch_rewtogo - V.detach() + ucb
            else:
                batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
                _,_,_, ucb = self.evaluate(batch_obs, batch_acts)
                Adv_k = self.compute_gae(batch_rews, batch_vals, batch_dones) + ucb
                V = self.critic(batch_obs).squeeze()
                batch_rewtogo = Adv_k + V.detach()

            # actual_time_step += np.sum(batch_lens)
            actual_iteration += 1
            
            # Normalize the advantage (to stabilize learning)
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)

            self.logger['actual_time_step'] = self.actual_time_step
            self.logger['actual_iteration'] = actual_iteration
            self.logger['learning_rate'] = self.actor_optim.param_groups[0]["lr"]
            
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches

            for _ in range(self.nb_epochs_per_iteration):
                
                if self.anneal_lr:
                    frac = (self.actual_time_step - 1.0) / total_timesteps
                    new_lr = self.lr * (1.0 - frac)
                    new_lr = max(new_lr, 0.0)
                    self.actor_optim.param_groups[0]["lr"] = new_lr
                    self.critic_optim.param_groups[0]["lr"] = new_lr
                
                np.random.shuffle(inds)
                
                for start in range(0, step, minibatch_size):
                    
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = Adv_k[idx]
                    mini_rewtogo = batch_rewtogo[idx]
                    
                    # Compute V_phi and pi_theta(a_t | s_t)
                    V, current_log_probs, entropy, _ = self.evaluate(mini_obs, mini_acts)

                    # Compute ratios between policies
                    ratios = torch.exp(current_log_probs - mini_log_prob)

                    # Actor surrogate losses
                    surrogate_loss1 = ratios * mini_advantage
                    surrogate_loss2 = torch.clamp(ratios, 1-self.clip, 1 + self.clip) * mini_advantage
                    
                    # Losses
                    actor_loss = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rewtogo) # or use F.mse_loss(V, batch_rewtogo)
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
            
            # Information at each iteration 
            self._log_summary()
            
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

            for timesteps_episode in range(self.max_timesteps_per_episode):
                
                self.actual_time_step += 1
                
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
                
                if self.continuous:
                    obs_key = tuple(int(np.round(o, decimals=2)*100) for o in obs) + (np.round(action[0], decimals=2),)
                    self.state_visit_count[obs_key] += 1
                else:
                    obs_key = tuple(int(np.round(o, decimals=2)*100) for o in obs) + (action,)
                    self.state_visit_count[obs_key] += 1
                
                if done:
                    break

            batch_lens.append(timesteps_episode + 1)
            batch_rews.append(ep_rews)
            if self.use_gae:
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)


            self.episode_rewards.append(np.sum(ep_rews))
            self.time_step_episode.append(self.actual_time_step)
            
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float) if self.continuous else torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewtogo = self.compute_rewtogo(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        
        if not self.use_gae:
            return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens
        else:
            return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

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
                noise = np.random.normal(0, self.noise_coef * self.env.action_space.high)
                action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)

            # Coloured noise addition
            else:
                noise = self.noise_array[self.actual_time_step % len(self.noise_array)]
                #print("noise = ", noise)
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
        white_noise = np.random.normal(0, self.noise_coef * self.env.action_space.high, size=size)
        frequency = np.fft.fftfreq(size[-1])
        amplitude = 1 / (np.abs(frequency) ** self.beta/2 + 1e-10)  # To avoid division by zero # use beta/2 is crucial for generating the correct type of 1/f^β noise. This scaling ensures that the noise has the desired power spectral density (PSD) characteristics
        amplitude[0] = 0
        noise_fft = np.fft.fft(white_noise)
        colored_noise_fft = noise_fft * amplitude
        colored_noise = np.fft.ifft(colored_noise_fft).real
        colored_noise *= 8e-11
        # print("color", colored_noise)
        return colored_noise
        
    def evaluate(self, batch_obs, batch_acts):
        # Query value V for each obs from critic nn
        V = self.critic(batch_obs).squeeze()

        # Log prob of batch actions using actor network
        mean = self.actor(batch_obs)
        
        #  Compute the UCB bonus
        if self.continuous:
            state_visit_count_tensor = torch.tensor([self.state_visit_count[tuple(int(np.round(o, decimals=2)*100) for o in obs) + (np.round(a, decimals=2),)] for obs, a in zip(batch_obs, batch_acts)])
        else:
            state_visit_count_tensor = torch.tensor([self.state_visit_count[tuple(int(np.round(o, decimals=2)*100) for o in obs) + (int(a),)] for obs, a in zip(batch_obs, batch_acts)])
        ucb_bonus = self.ucb_coef / torch.sqrt(state_visit_count_tensor + 1) # + 1 to ensure no division

        if self.continuous:
            distrib = MultivariateNormal(mean, self.cov_mat)
        else:
            action_probs = torch.nn.functional.softmax(mean, dim=-1)
            distrib = Categorical(action_probs)

        log_probs = distrib.log_prob(batch_acts)
        entropy = distrib.entropy()

        return V, log_probs, entropy, ucb_bonus

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
    
    def compute_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - int(ep_dones[t+1])) - ep_vals[t]
                    # delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lambda_gae * (1 - int(ep_dones[t])) * last_advantage
                # advantage = delta + self.gamma * self.lambda_gae * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

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

    def _init_hyperparameters(self, timesteps_per_batch = 4800, max_timesteps_per_episode = 1600, clip = 0.2,  ent_coef = 0.01, lr = 0.005 ,anneal_lr = False, noise_coef = 0.1, coloured_noise = False, beta = 0.5, use_gae=False, gamma = 0.95, lambda_gae = 0.95, ucb_coef = 0, num_minibatches = 4, render = False):
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode

        # Incorporate entropy, if coef = 0, as if no entropy incroporated. 0.01 is advised
        self.entropy_coef = ent_coef # A high coefficient penalized over deterministic policis --> more exploration. Useful for more complex environment
        
        # 0.2 is advised (paper)
        self.clip = clip

        # Possibility to use learning rate annealing
        self.anneal_lr = anneal_lr

        # Probably won't change
        self.nb_epochs_per_iteration = 3 # try to change it
        self.lr = lr # try to change it 3e-4 for the pendulum, 5e-3 for mountaincarcontinuous
        self.max_grad_norm = 1.0 # try to change it # add of cliping gradient for preventing exploding gradient --> more stable learning

        # Possibility to incorporate White and Coloured noise exploration
        self.beta = beta # 0.5 is advised
        self.coloured_noise = coloured_noise
        self.noise_coef = noise_coef
        # in the case of white noise discrete env : probability noise_coef to pick a full random action
        # in the case of white noise continuous env : sample from a gaussian with mean 0 and std dev being noise_coef % of the half possible action interval
        # for example if action space between -1 and 1, noise_coef = 0.1, then std dev of the gaussian is 0.1 (10 % of half interval)
        #             if action space between -2 and 2, noise_coef = 0.1, then std dev of the gaussian is 0.2 (10 % of half interval)
        # in the case of coloured noise continuous env : 

        
        # Use the generalized advantage estimation
        self.use_gae = use_gae
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        # UCB bonus coefficient
        self.ucb_coef = ucb_coef
        
        # Number of minibatches
        self.num_minibatches = num_minibatches
        
        # Use render
        self.render = render
        self.render_every_i = 20
        
    
        '''
        # Seed management
        self.seed = 42 # to set seed for reproductibility
        if self.seed is not None:
            assert(type(self.seed) == int)
            # set seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")'''
        
    def plot_rewards_time_steps(self,seed_rewards, seed_time_steps, individual = False):
        
        # Create empty lists to store the data
        all_times = []
        all_rewards = []
        all_seeds = []

        # Loop over each seed's rewards and time steps
        for seed, rewards in enumerate(seed_rewards):
            for episode, reward in enumerate(rewards):
                all_times.append(seed_time_steps[seed][episode])
                all_rewards.append(reward)
                all_seeds.append(seed)

        # Create the DataFrame
        df = pd.DataFrame({
            'Time': all_times,
            'Rewards': all_rewards,
            'Seed': all_seeds
        })
        df.to_csv("PPO_Pendulum.csv")
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        if individual:
            sns.lineplot(data=df, x='Time', y='Rewards', hue = "Seed", legend="full")
        else:
            sns.lineplot(data=df, x='Time', y='Rewards')
            
        plt.title('Rewards over episodes')
        plt.xlabel('Time steps')
        plt.ylabel('Rewards')
        plt.tight_layout()
        name = 'n_' + str(self.timesteps_per_batch)  + '_clip'  + str(self.clip) + '_ent' + str(self.entropy_coef) + '_lr' + str(self.lr) + '_anneal' + str(self.anneal_lr) + '_n' + str(self.noise_coef) + '_col' + str(self.coloured_noise) + '_beta' + str(self.beta) + '_gae' + str(self.use_gae) + '_gam' + str(self.gamma) + '_lam' + str(self.lambda_gae) + '_ucb' + str(self.ucb_coef) + 'batch' + str(self.num_minibatches)  + '.png'
        location = './plots/' + name
        plt.savefig(location)
        
    def plot_rewards_episodes(self, seed_rewards, individual = False):

        test_rewards = []
        test_episodes = []
        seeds = []

        for seed, rewards in enumerate(seed_rewards):
            for episode_idx, reward in enumerate(rewards):
                test_rewards.append(reward)
                test_episodes.append(episode_idx)
                seeds.append(seed)

        df = pd.DataFrame({
            'Episode': test_episodes,
            'Test_rewards': test_rewards,
            'Seed': seeds
        })
        df.to_csv("PPO_CartPole.csv")
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        if individual:
            sns.lineplot(data=df, x='Episode', y='Test_rewards', hue = "Seed", legend="full")
        else:
            sns.lineplot(data=df, x='Episode', y='Test_rewards')
            
        plt.title('Rewards over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.tight_layout()
        name = 'n_' + str(self.timesteps_per_batch)  + '_clip'  + str(self.clip) + '_ent' + str(self.entropy_coef) + '_lr' + str(self.lr) + '_anneal' + str(self.anneal_lr) + '_n' + str(self.noise_coef) + '_col' + str(self.coloured_noise) + '_beta' + str(self.beta) + '_gae' + str(self.use_gae) + '_gam' + str(self.gamma) + '_lam' + str(self.lambda_gae) + '_ucb' + str(self.ucb_coef) + 'batch' + str(self.num_minibatches)  + '.png'
        location = './plots/' + name
        plt.savefig(location)
        print('Plot saved at:', location)

    def generate_colored_noise(self):
        white = np.random.normal(size=(TOTAL_TIMESTEPS, self.act_dim))
        b = [0.02109238, 0.07113478, 0.68873558, -0.18234586, -0.10213203]
        a = [1, -0.131106, 0.20236, -0.0336, -0.0117]
        pink = np.zeros_like(white)
        for d in range(self.act_dim):
            pink[:, d] = lfilter(b, a, white[:, d])
        pink *= self.noise_coef
        return pink


def set_seed(env, seed):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

TOTAL_TIMESTEPS = 10000
if __name__ == "__main__":
    
    seed_rewards = []
    seed_time_steps = []
    for seed in [42]: #[42,380,479]
        print('Seed:', seed)
        
        env = gym.make('Acrobot-v1') # Possible env : Pendulum-v1 (continuous)/ CartPole-v1 (discrete) / MountainCarContinuous-v0 (continuous) / MountainCar-v0 (discrete)
        set_seed(env, seed)
        model = PPO(env)
        model._init_hyperparameters(timesteps_per_batch=300, max_timesteps_per_episode=500, clip=0.2, ent_coef=0.01, lr=0.005, anneal_lr=True, noise_coef=0.0, coloured_noise=False, beta=0, use_gae=True, gamma=0.99, lambda_gae=0.95, ucb_coef=0.001, num_minibatches=4, render=False)
        # model._init_hyperparameters(timesteps_per_batch=50, max_timesteps_per_episode=500, clip=0.2, ent_coef=0.01,lr=0.005, anneal_lr=True, noise_coef=0.0, coloured_noise=False, beta=0, use_gae=True, gamma=0.99, lambda_gae=0.95, ucb_coef=0.001, num_minibatches=4, render=False)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        
        seed_rewards.append(model.episode_rewards)
        
        # seed_time_steps.append(model.time_step_episode)
        
    model.plot_rewards_episodes(seed_rewards, individual = False)
    # model.plot_rewards_time_steps(seed_rewards, seed_time_steps, individual = False)