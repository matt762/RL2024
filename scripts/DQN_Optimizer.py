import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from .DQN import DQN
from .ReplayMemory import ReplayMemory, Transition


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class DQN_Optimizer(object):
    # Implementation of Deep Q learning based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html tutorial
    # with some adaptations to fit better the original algorithm from the paper 
    # "Playing Atari with Deep Reinforcement Learning, Mnih et al." https://arxiv.org/abs/1312.5602
    def __init__(self, env, replay_memory_size = 10000, param_dict = None, num_episodes = 50):
        # Initialize gymnasium environment
        self.env = env
        n_actions = env.action_space.n
        state, _ = env.reset()
        n_observations = len(state)

        # Initialize the DQnetwork 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                "tau" : 0.005,
                "learning_rate" : 1e-3,
                }
        else:
            self.param_dict = param_dict
        self.memory = ReplayMemory(replay_memory_size)
        self.steps_done = 0
        self.episode_durations = []

        self.num_episodes = num_episodes
        if ( not torch.cuda.is_available() and num_episodes > 50):
            print("The specified number of episodes might be big for optimization on cpu")
        self.episode_cumulative_reward = []
            
        


    def set_optimizer(self, optimizer: str = "adam"):
        if(optimizer == "adam"):
            self.optimizer = optim.AdamW(self.policy_network.parameters(), lr = self.param_dict.get("learning_rate"), amsgrad=True)
        elif (optimizer == "sgd"):
            self.optimizer = optim.SGD(self.policy_network.parameters(), lr = self.param_dict.get("learning_rate"))
        elif (optimizer == "RMSProp"):
            self.optimizer = optim.RMSprop(self.policy_network.parameters(),lr =self.param_dict.get("learning_rate"))
        else:
            print("invalid optimizer, please use : ...")


    def run_optimization(self):
        for i_episode in tqdm(range(self.num_episodes), desc="Current episode"):
            # Init env and get initial state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
            episode_reward = 0
            for t in count():
                # Select action and collect reward 
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward

                reward = torch.tensor([reward], device = self.device)
                done = terminated or truncated

                if terminated: 
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype= torch.float32, device = self.device).unsqueeze(0)

                # Save the transition in the replay memory
                self.memory.push(state,action, next_state, reward)

                state = next_state
                self.optimization_step()

                target_network_state_dict = self.target_network.state_dict()
                policy_network_state_dict = self.policy_network.state_dict()
                for key in policy_network_state_dict:
                    target_network_state_dict[key] = policy_network_state_dict[key] * self.param_dict.get("tau") + target_network_state_dict[key] * (1-self.param_dict.get("tau"))
                    self.target_network.load_state_dict(target_network_state_dict)

                if done:
                    self.episode_durations.append(t+1)
                    self.episode_cumulative_reward.append(episode_reward)
                    break

        print("Optimization complete")


    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.compute_epsilon()
        self.steps_done += 1
        if sample > eps_threshold:
            # exploit if >eps
            with torch.no_grad():
                return self.policy_network(state).max(1).indices.view(1,1)
        else:
            # explore otherwise
            return torch.tensor([[self.env.action_space.sample()]], device = self.device, dtype=torch.long)
        

    def optimization_step(self):
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

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def compute_epsilon(self):
        # Compute epsilon with a linear interpolation 
        if self.steps_done > self.param_dict.get("decay_steps"):
            return self.param_dict.get("eps_end")
        else:
            return self.param_dict.get("eps_start") - (self.param_dict.get("eps_start") - self.param_dict.get("eps_end"))*(self.steps_done/self.param_dict.get("decay_steps"))

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


    def plot_durations(self, moving_avg_width = 1):
        plt.figure()
        plt.title("Duration of episodes") 
        plt.xlabel("Episode") 
        plt.ylabel("Duration")
        plt.plot(self.episode_durations)

        if moving_avg_width > 1:
            # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
            cum_reward = np.cumsum(self.episode_durations)
            moving_average = (cum_reward[moving_avg_width:] - cum_reward[:-moving_avg_width]) / moving_avg_width
            plt.plot(moving_average)
            
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