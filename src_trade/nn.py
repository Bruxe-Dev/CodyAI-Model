import numpy as np


class NeuralNetwork:
    """3-layer network: input → 64 → 32 → output.  Adam + Leaky ReLU."""

    def __init__(self, input_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, 64) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(64)
        self.w2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32)
        self.w3 = np.random.randn(32, output_size) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(output_size)

        self._params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

        self.z1 = self.a1 = self.z2 = self.a2 = self.z3 = None

        total = self.w1.size + self.w2.size + self.w3.size
        print(f"   Network: {input_size}→64→32→{output_size}  |  {total:,} parameters")

    def _relu(self, x):  return np.where(x > 0, x, 0.01 * x)
    def _drelu(self, x): return np.where(x > 0, 1.0, 0.01)

    def forward(self, x):
        x = np.atleast_2d(x)
        self.z1 = x @ self.w1 + self.b1;       self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.w3 + self.b3
        return self.z3

    def backward(self, x, target, lr=0.001):
        x = np.atleast_2d(x); target = np.atleast_2d(target)
        err  = self.z3 - target
        loss = float(np.mean(err ** 2))
        d3   = 2 * err / err.size
        dw3  = self.a2.T @ d3;              db3 = d3.sum(0)
        d2   = (d3 @ self.w3.T) * self._drelu(self.z2)
        dw2  = self.a1.T @ d2;              db2 = d2.sum(0)
        d1   = (d2 @ self.w2.T) * self._drelu(self.z1)
        dw1  = x.T @ d1;                    db1 = d1.sum(0)

        grads = [dw1, db1, dw2, db2, dw3, db3]
        for g in grads: np.clip(g, -1, 1, out=g)

        self._t += 1
        b1, b2, ep = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self._t, 1 - b2 ** self._t
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            p -= lr * (self._m[i] / c1) / (np.sqrt(self._v[i] / c2) + ep)
        return loss