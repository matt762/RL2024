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
import seaborn

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
        self.state_visit_count = defaultdict(int)

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        actual_time_step = 0 # time_step simulated so far
        actual_iteration = 0 # iteration so far

        while actual_time_step < total_timesteps:
            
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

            actual_time_step += np.sum(batch_lens)
            actual_iteration += 1
            
            # Normalize the advantage (to stabilize learning)
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10)

            self.logger['actual_time_step'] = actual_time_step
            self.logger['actual_iteration'] = actual_iteration
            self.logger['learning_rate'] = self.actor_optim.param_groups[0]["lr"]
            
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches

            for _ in range(self.nb_epochs_per_iteration):
                
                if self.anneal_lr:
                    frac = (actual_time_step - 1.0) / total_timesteps
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
            
        return self.episode_rewards

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
                
                obs_key = tuple(int(np.round(o, decimals=2)*100) for o in obs)
                self.state_visit_count[obs_key] += 1
                
                if done:
                    break

            batch_lens.append(episode_t + 1)
            batch_rews.append(ep_rews)
            if self.use_gae:
                batch_vals.append(ep_vals)
                batch_dones.append(ep_dones)


            self.episode_rewards.append(np.sum(ep_rews))
            
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
        
        #  Compute the UCB bonus
        state_visit_count_tensor = torch.tensor([self.state_visit_count[tuple(int(np.round(o, decimals=2)*100) for o in obs)] for obs in batch_obs])
        ucb_bonus = self.ucb_coef / torch.sqrt(state_visit_count_tensor + 1)

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
                name = 'c' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + '_gae' + str(self.use_gae) + '_gamma' + str(self.gamma) + '_lambda' + str(self.lambda_gae) + '_ucb' + str(self.ucb_coef) + '_minibatches' + str(self.num_minibatches) + '_annlr' + str(self.anneal_lr) + 'col_n' + str(self.coloured_noise) + 'n_coef' + str(self.noise_coef) + 'beta' + str(self.beta) + '.png'
            else:
                name = 'c' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + str(self.entropy_coef) + '_gae' + str(self.use_gae) + '_gamma' + str(self.gamma) + '_lambda' + str(self.lambda_gae) + '_ucb' + str(self.ucb_coef) + '_minibatches' + str(self.num_minibatches) + '_annlr' + str(self.anneal_lr) + 'col_n' + str(self.coloured_noise) + 'n_coef' + str(self.noise_coef) + '.png'
        else:
            name = 'd' + '_clip' + str(self.clip) + '_ent' + str(self.entropy_coef) + str(self.entropy_coef) + '_gae' + str(self.use_gae) + '_gamma' + str(self.gamma) + '_lambda' + str(self.lambda_gae) + '_ucb' + str(self.ucb_coef) + '_minibatches' + str(self.num_minibatches) + '_annlr' + str(self.anneal_lr) + 'n_coef' + str(self.noise_coef) + '.png'
        location = './plots_pendulum/' + name
        plt.savefig(location)

    def _init_hyperparameters(self, timesteps_per_batch = 4800, max_timesteps_per_episode = 1600, clip = 0.2,  ent_coef = 0.01, anneal_lr = False, noise_coef = 0.1, coloured_noise = False, beta = 0.5, use_gae=False, gamma = 0.95, lambda_gae = 0.95, ucb_coef = 0, num_minibatches = 4, render = False):
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.render_every_i = 20

        # Incorporate entropy, if coef = 0, as if no entropy incroporated. 0.01 is advised
        self.entropy_coef = ent_coef # A high coefficient penalized over deterministic policis --> more exploration. Useful for more complex environment
        
        # 0.2 is advised (paper)
        self.clip = clip

        # Possibility to use learning rate annealing
        self.anneal_lr = anneal_lr

        # Probably won't change
        self.nb_epochs_per_iteration = 3 # try to change it
        self.lr = 0.005 # try to change it 3e-4 for the pendulum, 5e-3 for mountaincarcontinuous
        self.max_grad_norm = 0.5 # try to change it # add of cliping gradient for preventing exploding gradient --> more stable learning

        # Possibility to incorporate White and Coloured noise exploration
        self.beta = beta # 0.5 is advised
        self.coloured_noise = coloured_noise
        self.noise_coef = noise_coef
        
        # Use the generalized advantage estimation
        self.use_gae = use_gae
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        #UCB bonus coefficient
        self.ucb_coef = ucb_coef
        
        # Number of minibatches
        self.num_minibatches = num_minibatches
        
        # render
        self.render = render
        
        
    
        '''
        # Seed management
        self.seed = 42 # to set seed for reproductibility
        if self.seed is not None:
            assert(type(self.seed) == int)
            # set seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")'''


def set_seed(env):
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_rewards(rewards, moving_avg_width = 1):
    plt.figure()
    plt.title("Cumulative reward over episodes") 
    plt.xlabel("Episode") 
    plt.ylabel("Collected reward")
    plt.plot(rewards)

    if moving_avg_width > 1:
        # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
        cum_reward = np.cumsum(rewards)
        moving_average = (cum_reward[moving_avg_width:] - cum_reward[:-moving_avg_width]) / moving_avg_width
        plt.plot(moving_average)

    plt.show()

if __name__ == "__main__":
    for seed in [42]: #[42,380,479]
        env = gym.make('Pendulum-v1') # Possible env : Pendulum-v1 (continuous)/ CartPole-v1 (discrete) / MOuntainCarContinuous-v0 (continuous) / MountainCar-v0 (discrete)
        set_seed(env)
        model = PPO(env)
        model._init_hyperparameters(coloured_noise=False, beta=0.5, use_gae=False, ucb_coef=0, ent_coef=0, anneal_lr=True, render=False)
        rew = model.learn(5000)
        plot_rewards(rew)

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