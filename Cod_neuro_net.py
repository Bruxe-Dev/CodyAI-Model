import numpy as np 

class NeuralNetwork:
    def __init__(self,input_size, hidden_size, output_size):

        #Layer 1: Connects input to hidden neurons
        self.w1 = np.random.randn(input_size,hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)

        #Layer 2 : Connects hidden neurons to output
        self.w2 = np.random.randn(hidden_size,output_size) * 0.01
        self.b2  = np.zeros(output_size)

    def relu(self,x):
        return np.maximum(0,x)

    def relu_derivative(self,x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2

        return self.z2

    def backward (self, x, target, learning_rate = 0.001):
        output = self.forward(x)
        loss = np.mean((output - target) **2)

        d_output = 2 * (output- target)/ len(target) #This calculates how each neuron contributed to the loss ( Gradient)

        d_w2 = np.outer(self.a1,d_output) #The outer product
        d_b2 = d_output

        d_hidden  = np.dot(d_output, self.w2.T) #Move output error back into the hidden layer (Transposing the matrix)

        #Filter the error so that it passes through neurons that were active during the Forward pass

        d_hidden *= self.relu_derivative(self.z1) #Error will pass through neurons with 1

        d_w1 = np.outer(x, d_hidden)
        d_b1 = d_hidden

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