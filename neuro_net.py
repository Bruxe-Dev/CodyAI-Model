import numpy as np 

class NeuroNetwork:
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
    def predict(self, state):
        scores = self.forward(state)
        return np.argmax(scores)


if __name__== "__main__":
    net = NeuroNetwork(input_size=11,hidden_size=256,output_size=3)

    fake_state = np.random.randn(11)
    print("Game state (11 inputs):", np.round(fake_state, 2))

    scores = net.forward(fake_state)
    print("\nAction scores:", np.round(scores, 4))
    print("  0=turn left, 1=go straight, 2=turn right")

    action = net.predict(fake_state)
    print(f"\nChosen action: {action}")
    print("(This is random for now — it hasn't learned anything yet!)")