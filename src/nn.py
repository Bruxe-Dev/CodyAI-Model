import numpy as np 

class NeuralNetwork:
    def __init__(self,input_size, hidden_size, output_size):

        # Layer 1: Connects input to hidden neurons
        # Using He Initialization (optimal for ReLU activations)
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        # Layer 2: Connects hidden neurons to output
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2  = np.zeros(output_size)

        self.z1 = []
        self.a1 = []
        self.z2 = []
        self.output = []

    def relu(self,x):
        return np.maximum(0,x)

    def relu_derivative(self,x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2

        return self.z2

    def backward(self, x, target, learning_rate=0.001):
        # Ensure inputs are treated as 2D batch arrays
        x = np.atleast_2d(x)
        target = np.atleast_2d(target)

        output = self.forward(x)
        loss = np.mean((output - target) **2)

        # Gradient of MSE with respect to the output.
        # We divide by the number of output features (target.shape[1]), so that gradients 
        # naturally sum across the batch, preserving the same scale as the prior SGD approach.
        d_output = 2 * (output - target) / target.shape[1] 

        # Vectorized backpropagation using Matrix Multiplication (Dot Products)
        d_w2 = np.dot(self.a1.T, d_output) 
        d_b2 = np.sum(d_output, axis=0)

        d_hidden  = np.dot(d_output, self.w2.T) #Move output error back into the hidden layer (Transposing the matrix)

        # Filter the error so that it passes through neurons that were active during the Forward pass
        d_hidden *= self.relu_derivative(self.z1) # Error will pass through neurons with 1

        d_w1 = np.dot(x.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0)

        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2

        return loss
    def predict(self, state):
        scores = self.forward(state)
        return np.argmax(scores)

if __name__ == "__main__":
    net = NeuralNetwork(input_size=11, hidden_size=256, output_size=3)

    # Let's teach it one simple rule:
    # Given this state → action 2 (turn right) should score highest
    fake_state  = np.array([1,0,0,1,0,0,1,0,0,0,1], dtype=float)
    fake_target = np.array([0.0, 1.0, 0.0])  # we WANT output 2 to be 1.0

    print("Training the network on one example...\n")
    print(f"{'Step':<8} {'Loss':>10}  {'Scores (left, straight, right)'}")
    print("-" * 60)

    for step in range(1, 301):
        loss = net.backward(fake_state, fake_target, learning_rate=0.01)

        if step % 30 == 0 or step == 1:
            scores = net.forward(fake_state)
            print(f"{step:<8} {loss:>10.6f}  {np.round(scores, 3)}")

    print("\nFinal chosen action:", net.predict(fake_state))
    print("(Should be 1 = go straight )")