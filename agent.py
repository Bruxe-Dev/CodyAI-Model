import numpy as np 
import random
from collections import deque
from neuro_net import NeuralNetwork

def __init__ (self):
    self.input_size = 11
    self.hidden_size = 256
    self. output_size = 3

    self.network = NeuralNetwork(self.input_size,self.hidden_size,self.output_size)

    self.memory = deque (maxlen=100_000) #Deque will delete the memory when full

    self.gamma = 0.9 #Discount (Model focuses on future rewards)
    self.epsilon = 1.0 # Exploration rate (from 1.0 to 100% each try)
    self.epsilon_min = 0.01 # The agent must at least explore 1% for each try
    self.epsilon_decay = 0.995 #Ensures Exploration and Exploitation
    self.learning_rate = 0.001 # How the NN updates its bias and Weight
    self.batch_size = 64 #Learn from 64 past experiences at once


def remember (self, state, action, reward, next_state, done):
    self.memory.append((state,actio,reward,next_state,done))

def act (self, state):
    if random.random() < self.epsilon:
        return random.randint(0,2)

    return self.network.predict(state)

def learn(self):
    if len(self.memory) < self.batch_size:
        return
    
    batch = random.sample(self.memory,self.batch_size)

    for state, action, reward, next_state, done in batch:
        if done:
            q_target = reward
        else:
            future_q = np.max(self.network.forward(next_state))
            q_target = reward + self.gamma * future_q
            