import numpy as np 
import random 
from collections import deque
from src.nn import NeuralNetwork

class TradingDQNAgent:
    def __init__(self):
        self.input_size = 15
        self.hidden_size = 512
        self.output_size = 3

        self.network = NeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size
        )

        self.memory = deque (maxlen=100_000)

        self.gamma = 0.95 # Discount (Model focuses on future rewards) - Adjust this, higher = more focus on future rewards
        self.epsilon = 1.0 # Exploration rate (from 1.0 to 100% each try) - Adjust this, higher = more exploration
        self.epsilon_min = 0.01 # The agent must at least explore 1% for each try
        self.epsilon_decay = 0.995 # Ensures Exploration and Exploitation
        self.learning_rate = 0.0001 # How the NN updates its bias and Weight - better understanding that, it's the steps the gradient takes per update 😂
        self.batch_size = 64

