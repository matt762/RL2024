import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam

# step 3 : in paper do for 1,...N actor equivalent to collect a set of trajectories. One trajectory = 1 actor

class PPO:
    def __init__(self, env):
        # Hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Define actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Actor and critic optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

        # Create our variable for the matrix. 0.5 is arbitrary.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Cov matrix
        self.cov_mat = torch.diag(self.cov_var) 
        

    def learn(self, total_time_steps):
        actual_time_step = 0

        while actual_time_step < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()

            # Compute how many timesteps collected this batch
            actual_time_step += np.sum(batch_lens)

            # Compute V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Compute Advantage
            Adv_k = batch_rewtogo - V.detach()

            # Normalize advantages for better stability
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10) # Add 1e-10 to avoid div by 0

            for _ in range(self.nb_epochs_by_iteration):
                # Compute V_phi and pi_theta(a_t | s_t)
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)

                # Ratio
                ratios = torch.exp(current_log_probs - batch_log_probs)

                # Calculate surrogate losses (objective function that is optimized during training to update the parameters of the policy (actor) network)
                surrogate_loss1 = ratios * Adv_k
                surrogate_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * Adv_k # Clamps all elements in input into the range [ min, max ]

                # Actor and critic loss
                actor_loss = -(torch.min(surrogate_loss1, surrogate_loss2)).mean() # Objective function to optimize during training / - as we want to max but uses Adam opt which minimized loss
                critic_loss = nn.MSELoss()(V, batch_rewtogo)

                # Calculate gradients and performing backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()



    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rewtogo = []         # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        time_step_batch = 0

        # Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episodee
        ep_rews = []

        while time_step_batch < self.timesteps_per_batch:

            # Rewards collecter in this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for episode_t in range(self.max_timesteps_per_episode):     # 1) mais ducoup on a pas problème que là il le fait ici le time_step_batch += 1 ducoup si on dépasse le while il sera fait que après la boucle ???? ou bien ??????
                # Increment time step ran in this batch so far
                time_step_batch += 1

                # Collect observation
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # Break if environment tells episode is terminated
                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(episode_t + 1) # + 1 du to timestep starting at 0
            batch_rews.append(ep_rews)

        # Convert to tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewtogo = self.compute_rewtogo(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens


    def get_action(self, obs):
        # Query actor network for a mean action
        mean = self.actor(obs)
        # Create Multivariable Normal Distribution
        distrib = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = distrib.sample()
        log_prob = distrib.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    

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
        # Query critic network for a value V for each observations in the actual batch_observation
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network. This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
        

    def _init_hyperparameters(self):
        # Default values
        self.timesteps_per_batch = 4800 # nb of timestep collected for each batch :  if 0 trajectory (episode) converge, this will be 3 trajectories (episodes) by batch 3 * 1600
        self.max_timesteps_per_episode = 1600 # max nb of timestep per episode (1600 actions)
        # In our batch, we’ll be running episodes until we hit self.timesteps_per_batch timesteps
        self.gamma = 0.95
        # Nb of epochs
        self.nb_epochs_by_iteration = 5 # nb updates per iteration
        # Epsilon clip
        self.clip = 0.2 # as recommended by the paper
        # Learning rate of Adam
        self.lr = 0.005

#!pip install gym==0.25.2
#!pip install gym-notices==0.0.8
import gym
from gym import Env
# For visualization
#from gym.wrappers.monitoring import video_recorder
#from IPython.display import HTML
#from IPython import display
#import glob
#import matplotlib as plt
env = gym.make('Pendulum-v1', g=9.81)
env.seed(0)
model = PPO(env)
model.learn(1000)

'''
# Create the plot
fig = plt.figure(figsize=(20, 6))
ax = fig.add_subplot(111)

# Plot the scores with specified colors and labels
ax.plot(np.arange(1, len(scores_rwd2go) + 1), scores_rwd2go, color='green', label='No Baseline')

# Set the labels with a larger font size
ax.set_ylabel('Total reward (= time balanced)', fontsize=20)
ax.set_xlabel('Episode #', fontsize=20)

# Set the tick labels to a larger font size
ax.tick_params(axis='both', which='major', labelsize=15)

# Add a legend with a specified font size
ax.legend(fontsize=20)

# Show the plot
plt.show()
'''
