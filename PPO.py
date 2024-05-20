import torch
import torch.nn as nn
import numpy as np

from torch.distributions import MultivariateNormal
from network import FeedForwardNN
from torch.optim import Adam
import time
from torch.distributions import Categorical

# step 3 : in paper do for 1,...N actor equivalent to collect a set of trajectories. One trajectory = 1 actor

class PPO:
    def __init__(self, env):
        # Hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0] # for continuous env
        # self.act_dim = 2 # for discrete env just put nb nb act dim = nb possible discrete actions

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

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
			'delta_t': time.time_ns(),
			'actual_time_step': 0,          # timesteps so far
			'actual_iteration': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
        

    def learn(self, total_time_steps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_time_steps} timesteps")
        actual_time_step = 0
        actual_iteration = 0

        while actual_time_step < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens = self.rollout()

            # Compute how many timesteps collected this batch
            actual_time_step += np.sum(batch_lens)

            # Increment the number of iterations
            actual_iteration += 1

			# Logging timesteps so far and iterations so far
            self.logger['actual_time_step'] = actual_time_step
            self.logger['actual_iteration'] = actual_iteration


            # Compute V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Compute Advantage
            Adv_k = batch_rewtogo - V.detach()

            # Normalize advantages for better stability
            Adv_k = (Adv_k - Adv_k.mean()) / (Adv_k.std() + 1e-10) # Add 1e-10 to avoid div by 0

            for _ in range(self.nb_epochs_by_iteration):
                # Compute V_phi and pi_theta(a_t | s_t)
                #V, current_log_probs, entropy = self.evaluate(batch_obs, batch_acts) # ADD OF ENTROPY
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
                # Ratio
                ratios = torch.exp(current_log_probs - batch_log_probs)

                # Calculate surrogate losses (objective function that is optimized during training to update the parameters of the policy (actor) network)
                surrogate_loss1 = ratios * Adv_k
                surrogate_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * Adv_k # Clamps all elements in input into the range [ min, max ]

                # Actor and critic loss
                actor_loss = -(torch.min(surrogate_loss1, surrogate_loss2)).mean() # Objective function to optimize during training / - as we want to max but uses Adam opt which minimized loss
                #entropy_loss = entropy.mean() #ADD OF ENTROPY
                #actor_loss = actor_loss - self.entropy_coef * entropy_loss # Add entropy LOSS
                critic_loss = nn.MSELoss()(V, batch_rewtogo)

                # Calculate gradients and performing backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            self._log_summary()

            # Save our model if it's time
            if actual_iteration % self.save_frequency == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')



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

            max_timesteps = env.spec.max_episode_steps
            print(f"Maximum number of timesteps per episode: {max_timesteps}")              
            seed_info = self.env.seed()
            print("Seed information : ", seed_info)
            done = False

            for episode_t in range(self.max_timesteps_per_episode):
                    
                # If render is specified, render the environment
                if self.render and (self.logger['actual_iteration'] % self.render_every_i == 0) and len(batch_lens) == 0:
                #if self.render and (self.logger['actual_time_step'] % 1000 == 0): # CAPTE PAS TROP COMMENT LE RENDER MARCHE (ici pour le cartpole car lent sinon)
                    self.env.render()
                         
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

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rewtogo, batch_lens


    def get_action(self, obs):
        # Query actor network for a mean action
        mean = self.actor(obs)
        # Create Multivariable Normal Distribution
        distrib = MultivariateNormal(mean, self.cov_mat) # for continuous env
        #action_probs = torch.nn.functional.softmax(mean, dim=-1) # for discrete env
        #distrib = Categorical(action_probs) # for discrete env
        # Sample an action from the distribution and get its log prob
        action = distrib.sample() 
        log_prob = distrib.log_prob(action)

        return action.detach().numpy(), log_prob.detach() # for continuous env
        #return action.item(), log_prob.detach() # for discrete env
    

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
        distrib = MultivariateNormal(mean, self.cov_mat) # for continuous env
        log_probs = distrib.log_prob(batch_acts) # for continuous env

        #action_probs = torch.nn.functional.softmax(mean, dim=-1) # for discrete env
        #distrib = Categorical(action_probs) # for discrete env
        # Sample an action from the distribution and get its log prob

        return V, log_probs #, distrib.entropy()
        

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

        self.save_frequency = 100  # 10 is okey for mountain cars envs
        self.render = True
        self.render_every_i = 10 # 10 is okey for mountain cars envs
        #self.seed = 0

        #self.entropy_coef = 0.01

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

#!pip install gym==0.25.2
#!pip install gym-notices==0.0.8
import gym
from gym import Env
import gymnasium
# For visualization
#from gym.wrappers.monitoring import video_recorder
#from IPython.display import HTML
#from IPython import display
#import glob
#import matplotlib as plt
#env = gym.make('Pendulum-v1', g=9.81)
env = gym.make('MountainCarContinuous-v0')
#env.seed(0)
model = PPO(env)
model.learn(200000)

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
