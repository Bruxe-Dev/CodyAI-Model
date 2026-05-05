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
        
