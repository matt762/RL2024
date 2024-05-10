import torch
from torch.distributions import MultivariateNormal
import gymnasium as gym
from network import FeedForwardNN
from torch.distributions import MultivariateNormal


#jsp si besoin
from gym import Env
env = gym.make('Pendulum-v1')


# step 3 : in paper do for 1,...N actor equivalent to collect a set of trajectories. One trajectory = 1 actor

class PPO:
    def __init__(self, env : Env):
        # Hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Define actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Create our variable for the matrix. 0.5 is arbitrary.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Cov matrix
        self.cov_mat = torch.diag(self.cov_var) 
        

    def learn(self, total_time_steps):
        actual_time_step = 0

        while actual_time_step < total_time_steps:
            actual_time_step += 1

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()


    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rewtogo = []         # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        time_step_batch = 0

        while time_step_batch < self.timesteps_per_batch:

            # Rewards of this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for episode_t in range(self.max_timesteps_per_episode):
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

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(episode_t + 1) # + 1 du to timestep starting at 0
            batch_rews.append(ep_rews)

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

        # Sample an action from the distribution and get its log prob>
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
        

    def _init_hyperparameters(self):
            # Default values
            self.timesteps_per_batch = 4800 # nb of timestep collected for each batch :  if none trajectory (episode) converge, this will be 3 trajectories (episodes) by batch
            self.max_timesteps_per_episode = 1600 # max nb of timestep per episode (1600 actions)
            # In our batch, we’ll be running episodes until we hit self.timesteps_per_batch timesteps
            self.gamma = 0.95
