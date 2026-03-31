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