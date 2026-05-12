import numpy as np
import random
from collections import deque
from src_trade.nn import NeuralNetwork

class TradingDQNAgent:
    def __init__(self, input_size=31):
            self.input_size = input_size
            self.hidden_size = 512
            self.output_size = 3
            self.network = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            self.memory = deque(maxlen=100_000) 
            self.gamma = 0.95 
            self.epsilon = 1.0
            self.epsilon_min = 0.01 
            self.epsilon_decay = 0.9995 #
            self.learning_rate = 0.00025 
            self.batch_size = 64

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        q_values = self.network.forward(state)
        return np.argmax(q_values[0])

    def learn(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        if states.ndim == 3: states = states.reshape(self.batch_size, -1)
        if next_states.ndim == 3: next_states = next_states.reshape(self.batch_size, -1)

        q_values = self.network.forward(states)
        future_qs = self.network.forward(next_states)
        
        targets = q_values.copy()
        
        updates = rewards + self.gamma * np.max(future_qs, axis=1) * (1 - dones)
        targets[np.arange(self.batch_size), actions] = updates

        loss = self.network.backward(states, targets, self.learning_rate)
    
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
            
        return loss

    def save(self, filepath):
        np.savez(filepath, w1=self.network.w1, b1=self.network.b1, 
                 w2=self.network.w2, b2=self.network.b2, 
                 w3=self.network.w3, b3=self.network.b3)

    def load(self, filepath):
        data = np.load(filepath)
        self.network.w1, self.network.b1 = data["w1"], data["b1"]
        self.network.w2, self.network.b2 = data["w2"], data["b2"]
        self.network.w3, self.network.b3 = data["w3"], data["b3"]