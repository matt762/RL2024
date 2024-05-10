from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch

from gym import envs

class PPO:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        self._init_hyperparameters()
        
        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        
        self.gamma = 0.95
        
    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
        
    def rollout(self):
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        
        batch_masks = []  # Initialize masks
        batch_values = []  # Initialize values

        time_step = 0 # Timesteps run so far

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while time_step < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                # Compute mask and value
                mask = 0 if done else 1
                value = self.critic(torch.as_tensor(obs, dtype=torch.float32))

                # Track mask and value
                batch_masks.append(mask)
                batch_values.append(value.item())

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_masks, batch_values
    
    def learn(self, total_timesteps):
        time_spent = 0 # Timesteps simulated so far
        while time_spent < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Compute next value
            next_value = self.evaluate(batch_obs[-1])
            # Compute returns using GAE
            attention = self.compute_gae(next_value, batch_rtgs, batch_masks, batch_values)
            
            
    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        return V
    
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        # Gemeralized advantage estimation
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns