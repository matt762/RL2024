# Import packages
import sys
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from IPython.display import clear_output
from IPython import display

from eval_policy import eval_policy

device = torch.device("cpu")

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, activation, layers=[32,32,16]):
        super().__init__()

        # Define layers with ReLU activation
        self.linear1 = torch.nn.Linear(input_size, layers[0])
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(layers[0], layers[1])
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(layers[1], layers[2])
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(layers[2], output_size)
        self.output_activation = activation

        # Initialization using Xavier normal (a popular technique for initializing weights in NNs)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.linear3.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float)
        
        # Forward pass through the layers
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_activation(self.output_layer(x))
        return x

def compute_R2G_per_episode(batch_rews, gamma):
    # The rewards-to-go (rtg) per episode per batch to return
    batch_rtgs = []
    
    # Iterate through each episode backwards to maintain same order in batch_rtgs
    for ep_rews in reversed(batch_rews):
        discounted_reward = 0 # Discounted reward so far
        
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)
            
    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs

def generate_single_episode(env, actor):
    """
    Generates an episode by executing the current policy in the given env
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    max_t = 1000 # max horizon within one episode
    state, _ = env.reset()
        
    for t in range(max_t):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = actor.forward(Variable(state)) # get each action choice probability with the current policy network
        action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy())) # probablistic
        # action = np.argmax(probs.detach().numpy()) # greedy
        
        # compute the log_prob to use this in parameter update
        log_prob = torch.log(probs.squeeze(0)[action])
        
        # append values
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        
        # take a selected action
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated | truncated:
            break
            
    return states, actions, rewards, log_probs

def generate_multiple_episodes(env, actor, max_batch_size=500):
    """
    Generates an episode by executing the current policy in the given env
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    max_t = 1000 # max horizon within one episode
    i = 0
    
    while i < max_batch_size:
        state, _ = env.reset()
        reward_per_epi = []
        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = actor.forward(Variable(state)) # get each action choice probability with the current policy network
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy())) # probablistic
            # action = np.argmax(probs.detach().numpy()) # greedy
            
            # compute the log_prob to use this in parameter update
            log_prob = torch.log(probs.squeeze(0)[action])
            
            # append values
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            
            # take a selected action
            state, reward, terminated, truncated, _ = env.step(action)
            reward_per_epi.append(reward)
            
            i += 1

            if terminated | truncated:
                break
        rewards.append(reward_per_epi)
        
    return states, actions, rewards, log_probs

def evaluate_policy(env, actor):
    """
    Compute accumulative trajectory reward
    """
    _, _, rewards, _ = generate_single_episode(env, actor)
    return np.sum(rewards)


def train_PPO_multi_epi(env, actor, policy_optimizer, critic, value_optimizer, num_epochs, clip_val=0.2, gamma=0.99, max_batch_size=100, entropy_coef=0.1, normalize_ad=True, add_entropy=True):

    # Generate an episode with the current policy network
    states, actions, rewards, log_probs = generate_multiple_episodes(env, actor, max_batch_size=max_batch_size)
    T = len(states)
    
    # Create tensors
    states = np.vstack(states).astype(float)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).view(-1,1).to(device)
    log_probs = torch.FloatTensor(log_probs).view(-1,1).to(device)

    # Compute total discounted return at each time step in each episode
    Gs = compute_R2G_per_episode(rewards, gamma).view(-1,1)
    
    # Compute the advantage
    states = states.to(device)
    state_vals = critic(states).to(device)
    with torch.no_grad():
        A_k = Gs - state_vals
    if normalize_ad:
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages
        
    for _ in range(num_epochs):
        V = critic(states)
        
        # Calculate probability of each action under the updated policy
        probs = actor.forward(states).to(device)
                
        # compute the log_prob to use it in parameter update
        curr_log_probs = torch.log(torch.gather(probs, 1, actions)) # Use torch.gather(A,1,B) to select columns from A based on indices in B
        
        # Calculate ratios r(theta)
        ratios = torch.exp(curr_log_probs - log_probs)
        
        # Calculate two surrogate loss terms in cliped loss
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-clip_val, 1+clip_val) * A_k
        
        # Caluculate entropy
        entropy = 0
        if add_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            entropy = torch.tensor([[e] for e in entropy])
        
        # Calculate clipped loss value
        actor_loss = (-torch.min(surr1, surr2) - entropy_coef * entropy).mean() # Need negative sign to run Gradient Ascent
        
        # Update policy network
        policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        # Update value net
        critic_loss = nn.MSELoss()(V, Gs)
        value_optimizer.zero_grad()
        critic_loss.backward()
        value_optimizer.step()        
        
    return actor, critic

# Create the environment.
env_name = "CartPole-v1"
#env = gym.make(env_name, render_mode='human')
env = gym.make(env_name)
nA = env.action_space.n
nS = 4

def train(env, nA, nS):
    # Define parameter values
    num_train_ite = 1000
    num_seeds = 5 # fit model with 5 different seeds and plot average performance of 5 seeds
    num_epochs = 10 # how many times we iterate the entire training dataset passing through the training
    eval_freq = 50 # run evaluation of policy at each eval_freq trials
    eval_epi_index = num_train_ite//eval_freq # use to create x label for plot
    returns = np.zeros((num_seeds, eval_epi_index))
    gamma = 0.99 # discount factor
    clip_val = 0.2 # hyperparameter epsilon in clip objective

    # Define parameter values
    returns = np.zeros((num_seeds, eval_epi_index))
    max_batch_size = 100
    entropy_coef = 0.1
    normalize_ad = True
    add_entropy = True

    policy_lr = 5e-4 # policy network's learning rate 
    baseline_lr = 1e-4

    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []

        # Define policy and value networks
        actor = NeuralNet(nS, nA, torch.nn.Softmax())
        actor_optimizer = optim.Adam(actor.parameters(), lr=policy_lr)
        critic = NeuralNet(nS, 1, torch.nn.ReLU())
        critic_optimizer = optim.Adam(critic.parameters(), lr=baseline_lr)
        
        for m in range(num_train_ite):
            # Train networks with PPO
            actor, critic = train_PPO_multi_epi(env, actor, actor_optimizer, critic, critic_optimizer, num_epochs, clip_val=clip_val, gamma=gamma, max_batch_size=max_batch_size, entropy_coef=entropy_coef, normalize_ad=normalize_ad, add_entropy=add_entropy)
            if m % eval_freq == 0:
                print("Episode: {}".format(m))
                G = np.zeros(20)
                for k in range(20):
                    g = evaluate_policy(env, actor)
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print("The avg. test reward for episode {0} is {1} with std of {2}.".format(m, reward_mean, reward_sd))
                reward_means.append(reward_mean)
                
                torch.save(actor.state_dict(), './ppo_actor.pth')
                torch.save(critic.state_dict(), './ppo_critic.pth')
                
        returns[i] = np.array(reward_means)

        
    # Plot the performance over iterations
    x = np.arange(eval_epi_index)*eval_freq
    avg_returns = np.mean(returns, axis=0)
    max_returns = np.max(returns, axis=0)
    min_returns = np.min(returns, axis=0)

    plt.fill_between(x, min_returns, max_returns, alpha=0.1)
    plt.plot(x, avg_returns, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    plt.title("PPO Learning Curve", fontsize = 24)
    plt.show()

def test(env, nA, nS):
    
    print('Testing PPO')

	# Build our policy the same way we build our actor model in PPO
    policy = NeuralNet(nS, nA, torch.nn.Softmax())
    
	# Load in the actor model saved by the PPO algorithm
    actor_model = 'ppo_actor.pth'
    policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

# Read the argument in input of the program
args = sys.argv
if len(args) > 1:
    if args[1] == 'train':
        train(env, nA, nS)
    elif args[1] == 'test':
        test(env, nA, nS)
else:
    print("No input argument provided.")