import random
from collections import namedtuple, deque


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, N):
        self.memory = deque([], maxlen=N)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Sample without replacement
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    



def main():
    memory = ReplayMemory(N = 10)

if __name__ == "__main__":
    main()