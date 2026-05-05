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

    
    def remember (self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self, state):

        # Explore: random action
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # HOLD, BUY, or SELL
        
        # Exploit: use neural network
        return self.network.predict(state)
    
    def learn(self):
        """
        Learn from past experiences (same as snake!)
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Calculate Q-values
        q_values = self.network.forward(states)
        future_qs = self.network.forward(next_states)
        
        # Update targets with Bellman equation
        targets = q_values.copy()
        updates = rewards + self.gamma * np.max(future_qs, axis=1) * (1 - dones)
        targets[np.arange(self.batch_size), actions] = updates
        
        # Train network
        self.network.backward(states, targets, self.learning_rate)
        
        # Decay epsilon (explore less over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath="trading_model.npz"):
        """Save model weights"""
        np.savez(filepath,
                 w1=self.network.w1, b1=self.network.b1,
                 w2=self.network.w2, b2=self.network.b2)
        print(f"✓ Trading model saved to {filepath}")
    
    def load(self, filepath="trading_model.npz"):
        """Load model weights"""
        data = np.load(filepath)
        self.network.w1 = data["w1"]
        self.network.b1 = data["b1"]
        self.network.w2 = data["w2"]
        self.network.b2 = data["b2"]
        print(f" Trading model loaded from {filepath}")
