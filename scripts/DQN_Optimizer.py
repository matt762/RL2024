import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.notebook import tqdm

from .DQN import DQN
from .ReplayMemory import ReplayMemory, Transition


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class DQN_Optimizer(object):
    # Implementation of Deep Q learning based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html tutorial
    # with some adaptations to fit better the original algorithm from the paper 
    # "Playing Atari with Deep Reinforcement Learning, Mnih et al." https://arxiv.org/abs/1312.5602
    def __init__(self, env, seed, param_dict = None):
        # Initialize gymnasium environment
        self.env = env

        # Define seeds
        state, _ = env.reset(seed=seed)
        self.set_seed(seed=seed)

        n_actions = env.action_space.n
        n_observations = len(state)



        # Initialize the double DQnetwork 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("pytorch will run on {}".format(self.device))
        self.policy_network = DQN(n_observations, n_actions).to(self.device)
        self.target_network = DQN(n_observations, n_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        if param_dict == None:
            self.param_dict = {
                "batch_size" : 32,
                "gamma" : 0.99,
                "eps_start" : 1,
                "eps_end" : 0.1,
                "decay_steps" : 1000,
                "learning_rate" : 1e-3,
                "train_episodes" : 200,
                "test_episodes" : 20,
                "replay_memory_size" : 100000,
                "optimizer" : "adam",
                "fill_rp_memory" : True,
                }
        else:
            self.param_dict = param_dict
        self.memory = ReplayMemory(self.param_dict.get("replay_memory_size"))
        self.steps_done = 0
        self.episode_durations = []
        self.epsilons = []
        self.epsilon = param_dict.get("eps_start")
        self.set_optimizer(self.param_dict.get("optimizer"))
        if ( not torch.cuda.is_available() and param_dict.get("train_episodes") > 50):
            print("The specified number of episodes might be big for optimization on cpu")
        self.episode_cumulative_reward = []

        self.epsilon = param_dict.get("eps_start")
            
        
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_optimizer(self, optimizer: str = "adam"):
        if(optimizer == "adam"):
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr = self.param_dict.get("learning_rate"))
        elif (optimizer == "sgd"):
            self.optimizer = optim.SGD(self.policy_network.parameters(), lr = self.param_dict.get("learning_rate"))
        elif (optimizer == "RMSProp"):
            self.optimizer = optim.RMSprop(self.policy_network.parameters(),lr =self.param_dict.get("learning_rate"))
        else:
            print("invalid optimizer, please use : ...")


    def run_optimization(self):
        if self.param_dict.get("fill_rp_memory"):
            print("Filling buffer memory...")
            self.fill_buffer_memory()
            print("Buffer memory filled, size: {}".format(len(self.memory)))
        mean_rewards_test = []
        train_progress = tqdm(range(self.param_dict.get("train_episodes")))
        for i_episode in train_progress:
            # Init env and get initial state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
            episode_reward = 0
            for t in count():
                train_progress.set_description("Current train episode eps : {:.2f}".format(self.epsilon))
                # Select action and collect reward 
                action = self.select_eps_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                self.steps_done += 1
                reward = torch.tensor([reward], device = self.device)

                if terminated: 
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype= torch.float32, device = self.device).unsqueeze(0)

                # Save the transition in the replay memory
                self.memory.push(state,action, next_state, reward)

                state = next_state
                self.minibatch_update()


                if terminated or truncated:
                    self.episode_durations.append(t+1)
                    self.episode_cumulative_reward.append(episode_reward)
                    self.epsilons.append(self.epsilon)
                    break
            self.update_epsilon()

            ## At the end of 10 episodes, run 20 episodes for test
            if i_episode % 10 == 0:
                test_rewards = self.test()
                reward_mean = test_rewards.mean()
                mean_rewards_test.append(reward_mean)
        print("Optimization complete")
        return mean_rewards_test


    def select_eps_greedy_action(self,state,eps = None):
        sample = random.random()
        if eps == None:
            eps_threshold = self.epsilon
        else:
            eps_threshold = eps
        if sample > eps_threshold:
            # exploit if >eps
            with torch.no_grad():
                return torch.argmax(self.policy_network(state)).reshape(1,1)
        else:
            # explore otherwise
            return torch.tensor([[self.env.action_space.sample()]], device = self.device, dtype=torch.long)
        

    def select_random_action(self):
        return torch.tensor(self.env.action_space.sample(), device = self.device).reshape(1,1)

    def fill_buffer_memory(self):
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device= self.device).unsqueeze(0)
        for i in range(self.memory.size):
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device = self.device).unsqueeze(0)
            
            action = self.select_random_action()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device = self.device)
            

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device = self.device).unsqueeze(0)

            self.memory.push(state, action, next_state, reward)
            state = next_state

    def minibatch_update(self):
        if len(self.memory) < self.param_dict.get("batch_size"):
            return
        transitions = self.memory.sample(self.param_dict.get("batch_size"))
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_network(state_batch).gather(1,action_batch)
        next_state_values = torch.zeros(self.param_dict.get("batch_size"), device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values*self.param_dict["gamma"] + reward_batch)

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        if self.steps_done%self.param_dict.get("steps_between_updates") == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def update_epsilon(self):
        # Update epsilon decay using a product

        if self.epsilon > self.param_dict.get("eps_end"):
            self.epsilon = self.epsilon*self.param_dict.get("eps_decay")
        else:
            self.epsilon = self.param_dict.get("eps_end")
        return self.epsilon

    def test(self):
        num_test_episodes = self.param_dict.get("test_episodes")
        rewards_test = np.zeros(num_test_episodes)
        test_progress = tqdm(range(num_test_episodes),leave=False)
        for i in test_progress:
            state, _ = self.env.reset()            
            episode_rewards = 0
            for t in count():
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.select_eps_greedy_action(state, eps=0)
                state, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_rewards += reward
                if terminated or truncated:
                    break
            rewards_test[i] = episode_rewards
            test_progress.set_description("Current test episode reward: {}".format(episode_rewards))
        return rewards_test


    def plot_rewards(self, moving_avg_width = 1):
        plt.figure()
        plt.title("Cumulative reward over episodes") 
        plt.xlabel("Episode") 
        plt.ylabel("Collected reward")
        plt.plot(self.episode_cumulative_reward)

        if moving_avg_width > 1:
            # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
            cum_reward = np.cumsum(self.episode_cumulative_reward)
            moving_average = (cum_reward[moving_avg_width:] - cum_reward[:-moving_avg_width]) / moving_avg_width
            plt.plot(moving_average)

        plt.show()


    def plot_epsilons(self):
        plt.figure()
        plt.title("Epsilon") 
        plt.xlabel("Episode") 
        plt.ylabel("Value")
        plt.plot(self.epsilons)
        plt.show()

    def plot_avg_step_reward(self, moving_avg_width = 1):
        plt.figure()
        plt.title("Average reward over episode steps") 
        plt.xlabel("Episode") 
        plt.ylabel("Average reward")
        avg_ep_reward = [self.episode_cumulative_reward[i]/self.episode_durations[i] for i in range(len(self.episode_durations))]
        plt.plot(avg_ep_reward)

        if moving_avg_width > 1:
            # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
            cum_reward = np.cumsum(avg_ep_reward)
            moving_average = (cum_reward[moving_avg_width:] - cum_reward[:-moving_avg_width]) / moving_avg_width
            plt.plot(moving_average)
            
        plt.show()

        

if __name__ == "__main__":
    pass                                                                                                                    