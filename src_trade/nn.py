import numpy as np


class NeuralNetwork:
    """
    3-layer fully-connected network  (input → 128 → 64 → output)
    Leaky ReLU activations + Adam optimiser.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        h2 = hidden_size // 2
        self.w1 = np.random.randn(input_size,  hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, h2)          * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(h2)
        self.w3 = np.random.randn(h2,          output_size) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_size)

        self._params   = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self._m        = [np.zeros_like(p) for p in self._params]
        self._v        = [np.zeros_like(p) for p in self._params]
        self._t        = 0
        self._b1_adam  = 0.9
        self._b2_adam  = 0.999
        self._eps_adam = 1e-8

        # Activation cache — written by forward(), read by backward()
        self.z1 = self.a1 = None
        self.z2 = self.a2 = None
        self.z3 = None

    def _lrelu(self, x):      return np.where(x > 0, x, 0.01 * x)
    def _lrelu_grad(self, x): return np.where(x > 0, 1.0, 0.01)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(x,       self.w1) + self.b1;  self.a1 = self._lrelu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2;  self.a2 = self._lrelu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        return self.z3

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=-1)

    def backward(self, x: np.ndarray, target: np.ndarray,
                 learning_rate: float = 0.001) -> float:
        """Uses cached activations — caller must call forward() first."""
        x      = np.atleast_2d(x)
        target = np.atleast_2d(target)

        error = self.z3 - target
        loss  = float(np.mean(np.square(np.clip(error, -1e3, 1e3))))
        n     = target.shape[0] * target.shape[1]
        d_out = 2.0 * error / n

        d_w3 = np.dot(self.a2.T, d_out);            d_b3 = np.sum(d_out, axis=0)
        d_h2 = np.dot(d_out, self.w3.T) * self._lrelu_grad(self.z2)
        d_w2 = np.dot(self.a1.T, d_h2);             d_b2 = np.sum(d_h2, axis=0)
        d_h1 = np.dot(d_h2, self.w2.T) * self._lrelu_grad(self.z1)
        d_w1 = np.dot(x.T, d_h1);                   d_b1 = np.sum(d_h1, axis=0)

        grads = [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]
        for g in grads:
            np.clip(g, -1.0, 1.0, out=g)

        self._t += 1
        bc1 = 1.0 - self._b1_adam ** self._t
        bc2 = 1.0 - self._b2_adam ** self._t
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = self._b1_adam * self._m[i] + (1.0 - self._b1_adam) * g
            self._v[i] = self._b2_adam * self._v[i] + (1.0 - self._b2_adam) * g * g
            p -= learning_rate * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + self._eps_adam)

        self._params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        return loss

    @property
    def weights(self): return [self.w1, self.w2, self.w3]
    @property
    def biases(self):  return [self.b1, self.b2, self.b3]