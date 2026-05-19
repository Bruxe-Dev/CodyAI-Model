import numpy as np

class NeuralNetwork:

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dropout_rate: float = 0.2):
        """
        Args
        ----
        input_size   : number of input features (28)
        hidden_sizes : list of hidden layer sizes e.g. [256, 128, 64]
        output_size  : number of actions (3)
        dropout_rate : fraction of neurons dropped during training (0 = off)
        """
        self.dropout_rate = dropout_rate
        self.training     = True   # set to False during act() / live inference

        # Build weight matrices for arbitrary depth
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

        self.n_layers = len(self.weights)

        # Adam state for every parameter matrix
        self._mw = [np.zeros_like(w) for w in self.weights]
        self._vw = [np.zeros_like(w) for w in self.weights]
        self._mb = [np.zeros_like(b) for b in self.biases]
        self._vb = [np.zeros_like(b) for b in self.biases]
        self._t  = 0
        self._b1 = 0.9
        self._b2 = 0.999
        self._ep = 1e-8

        # Activation cache
        self._z    = [None] * self.n_layers   # pre-activations
        self._a    = [None] * self.n_layers   # post-activations
        self._mask = [None] * (self.n_layers - 1)  # dropout masks

        # Count and report parameters
        total = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        arch  = f"{input_size} → " + " → ".join(str(s) for s in hidden_sizes) + f" → {output_size}"
        print(f"   Network: {arch}")
        print(f"   Parameters: {total:,}  ({total*4/1024:.1f} KB)")

    # ── activation ────────────────────────────────────────────────────────────
    def _lrelu(self, x):      return np.where(x > 0, x, 0.01 * x)
    def _lrelu_grad(self, x): return np.where(x > 0, 1.0,  0.01)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(x)
        for i in range(self.n_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self._z[i] = z
            if i < self.n_layers - 1:
                # Hidden layer: Leaky ReLU + optional dropout
                a = self._lrelu(z)
                if self.training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32)
                    mask /= (1.0 - self.dropout_rate)   # inverted dropout scaling
                    a   *= mask
                    self._mask[i] = mask
                else:
                    self._mask[i] = np.ones_like(a)
            else:
                # Output layer: linear (Q-values can be any sign/magnitude)
                a = z
            self._a[i] = a
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.training = False
        result = np.argmax(self.forward(x), axis=-1)
        self.training = True
        return result

    # ── backward ──────────────────────────────────────────────────────────────
    def backward(self, x: np.ndarray, target: np.ndarray,
                 learning_rate: float = 0.001) -> float:
        """
        Uses cached activations from the most recent forward() call.
        Does NOT re-run forward().
        """
        x      = np.atleast_2d(x)
        target = np.atleast_2d(target)
        output = self._a[-1]

        error  = output - target
        loss   = float(np.mean(np.square(np.clip(error, -1e3, 1e3))))
        n      = target.shape[0] * target.shape[1]

        # Output delta
        delta = 2.0 * error / n

        dw_list = []
        db_list = []

        for i in reversed(range(self.n_layers)):
            a_prev = x if i == 0 else self._a[i - 1]
            dw     = np.dot(a_prev.T, delta)
            db     = np.sum(delta, axis=0)
            dw_list.insert(0, dw)
            db_list.insert(0, db)

            if i > 0:
                # Backprop through activation + dropout
                delta = np.dot(delta, self.weights[i].T)
                delta *= self._mask[i - 1]               # dropout mask
                delta *= self._lrelu_grad(self._z[i - 1])

        # Gradient clipping
        for dw, db in zip(dw_list, db_list):
            np.clip(dw, -1.0, 1.0, out=dw)
            np.clip(db, -1.0, 1.0, out=db)

        self._t += 1
        bc1 = 1.0 - self._b1 ** self._t
        bc2 = 1.0 - self._b2 ** self._t

        for i, (dw, db) in enumerate(zip(dw_list, db_list)):
            self._mw[i] = self._b1 * self._mw[i] + (1 - self._b1) * dw
            self._vw[i] = self._b2 * self._vw[i] + (1 - self._b2) * dw * dw
            self._mb[i] = self._b1 * self._mb[i] + (1 - self._b1) * db
            self._vb[i] = self._b2 * self._vb[i] + (1 - self._b2) * db * db
            self.weights[i] -= learning_rate * (self._mw[i]/bc1) / (np.sqrt(self._vw[i]/bc2) + self._ep)
            self.biases[i]  -= learning_rate * (self._mb[i]/bc1) / (np.sqrt(self._vb[i]/bc2) + self._ep)

        return loss