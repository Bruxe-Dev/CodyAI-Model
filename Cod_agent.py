import numpy as np 
import random
from collections import deque
from neuro_net import NeuralNetwork

class DQNAgent:
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
        self.memory.append((state,action,reward,next_state,done))

    def act (self, state):
        if random.random() < self.epsilon:
            return random.randint(0,2)

        return self.network.predict(state)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
    
        batch = random.sample(self.memory, self.batch_size)

        # Create batched NumPy arrays
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Predict Q-values for current states and next states (Batched!)
        q_values = self.network.forward(states)
        future_qs = self.network.forward(next_states)

        targets = q_values.copy()

        # Update targets with the Q-learning formula
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(future_qs[i])

        # Perform one single vectorized backward pass for the whole batch
        self.network.backward(states, targets, self.learning_rate)

        # Decay epsilon — become less random over time (happens once per learning step)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath="snake_brain.npz"):
        np.savez(filepath,
                 w1=self.network.w1, b1=self.network.b1,
                 w2=self.network.w2, b2=self.network.b2)
        print(f"Model saved to {filepath}")

    def load(self, filepath="snake_brain.npz"):
        data = np.load(filepath)
        self.network.w1 = data["W1"]
        self.network.b1 = data["b1"]
        self.network.w2 = data["W2"]
        self.network.b2 = data["b2"]
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    agent = DQNAgent()

    print("Testing agent with fake experiences...\n")

    # Simulate 200 random game steps
    for i in range(200):
        state      = np.random.randn(11)
        action     = agent.act(state)
        reward     = random.choice([-10, 1, 10])
        next_state = np.random.randn(11)
        done       = random.random() < 0.05   # 5% chance of dying

        agent.remember(state, action, reward, next_state, done)

    # Now try to learn from those memories
    agent.learn()

    print(f"Memories stored : {len(agent.memory)}")
    print(f"Epsilon now     : {agent.epsilon:.4f}  (was 1.0)")
    print(f"Test action     : {agent.act(np.random.randn(11))}")
    print("\nAgent is working correctly!")