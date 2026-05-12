import numpy as np 

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Layer 1: Input → Hidden1
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        
        # Layer 2: Hidden1 → Hidden2
        self.hidden2_size = hidden_size // 2
        self.w2 = np.random.randn(hidden_size, self.hidden2_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(self.hidden2_size)
        
        # Layer 3: Hidden2 → Output
        self.w3 = np.random.randn(self.hidden2_size, output_size) * np.sqrt(2.0 / self.hidden2_size)
        self.b3 = np.zeros(output_size)

        # For gradient clipping convenience
        self.weights = [self.w1, self.w2, self.w3]
        self.biases = [self.b1, self.b2, self.b3]

        self.z1, self.a1 = None, None
        self.z2, self.a2 = None, None
        self.z3 = None

    def relu(self, x):
        return np.where(x > 0, x, x * 0.01) # Leaky ReLU to prevent dead neurons

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def forward(self, x):
        # Layer 1
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Layer 3 (Output)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        return self.z3

    def predict(self, x):
        scores = self.forward(x)
        return np.argmax(scores, axis=-1)

    def backward(self, x, target, learning_rate=0.001):
        x = np.atleast_2d(x)
        target = np.atleast_2d(target)
        
        output = self.forward(x)
        
        # 1. Calculate Error (Loss)
        # We use clip to prevent square of huge numbers causing overflow
        error = output - target
        loss = np.mean(np.square(np.clip(error, -1e3, 1e3)))
        
        # 2. Backpropagate through 3 layers
        d_output = 2 * error / (target.shape[0] * target.shape[1])
        
        # Layer 3 gradients
        d_w3 = np.dot(self.a2.T, d_output)
        d_b3 = np.sum(d_output, axis=0)
        
        # Backprop to layer 2
        d_hidden2 = np.dot(d_output, self.w3.T)
        d_hidden2 *= self.relu_derivative(self.z2)
        
        # Layer 2 gradients
        d_w2 = np.dot(self.a1.T, d_hidden2)
        d_b2 = np.sum(d_hidden2, axis=0)
        
        # Backprop to layer 1
        d_hidden1 = np.dot(d_hidden2, self.w2.T)
        d_hidden1 *= self.relu_derivative(self.z1)
        
        # Layer 1 gradients
        d_w1 = np.dot(x.T, d_hidden1)
        d_b1 = np.sum(d_hidden1, axis=0)

        # 3. GRADIENT CLIPPING (The Fix)
        # This prevents the "Mathematical Explosion" (Overflow)
        for grad in [d_w1, d_w2, d_w3, d_b1, d_b2, d_b3]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # 4. Update Weights
        self.w3 -= learning_rate * d_w3
        self.b3 -= learning_rate * d_b3
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        
        return loss