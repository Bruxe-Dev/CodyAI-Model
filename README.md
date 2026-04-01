# CodyAI-Model

A Snake game AI trained using Deep Q-Learning (DQN) with a custom neural network implementation from scratch (NumPy only).

## Project Structure

```
CodyAI-Model/
├── src/
│   ├── nn.py          # 2-layer neural network (ReLU, He init)
│   ├── dqn_agent.py   # DQN agent with experience replay
│   ├── snake_env.py   # Gym-like environment (11-dim state)
│   ├── train.py       # Training loop & visualization
│   └── __init__.py
├── game.py            # Human-playable Snake game (pygame)
├── snake_brain.npz    # Pre-trained model - Generated on training
├── Cod_brain.npz      # Alternate trained model
└── requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
```

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pandas`, `pygame`

## Usage

### Train the model

```bash
python -m src.train
```

The AI will train for 1000 games. Best model is auto-saved to `main_model.npz`.

Options:
- Edit `train.py` line 39: Set `render=False` for headless training (much faster)

### Watch Trained model Play

```bash
python -m src.train watch
```

Loads `main_model.npz` and displays the AI playing with pygame visualization.

### Play Human Game

```bash
python game.py
```

Controls: Arrow keys or WASD, SPACE to pause, ESC to quit.

## How It Works

### Neural Network Architecture (`src/nn.py`)
- **Input**: 11 neurons (danger sensors, direction, food direction)
- **Hidden**: 256 neurons (ReLU activation, He initialization)
- **Output**: 3 neurons (Q-values for [left, straight, right])
- Loss: MSE | Optimizer: SGD with vectorized gradients

### State Representation (11-dim)
```python
[ danger_straight, danger_right, danger_left,    # 3 collision sensors
  dir_up, dir_down, dir_left, dir_right,        # 4 direction (one-hot)
  food_up, food_down, food_left, food_right ]    # 4 food direction
```

### Action Space
- **0**: Turn left
- **1**: Go straight
- **2**: Turn right

### Reward Structure (`src/snake_env.py`)
| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Survive | +1 |
| Collision / Timeout | -10 |

### DQN Hyperparameters (`src/dqn_agent.py`)
| Param | Value |
|-------|-------|
| Gamma | 0.9 |
| Epsilon start | 1.0 |
| Epsilon min | 0.01 |
| Epsilon decay | 0.995 |
| Batch size | 64 |
| Replay buffer | 100,000 |
| Learning rate | 0.001 |

## Model Persistence

Models are saved as `.npz` files containing weight matrices:
```python
# Save (auto-triggered on new best score)
agent.save("main_model.npz")

# Load (in watch mode)
agent.load("main_model.npz")
```

## Files

| File | Description |
|------|-------------|
| `src/nn.py` | Custom 2-layer neural network with backprop |
| `src/dqn_agent.py` | Agent with experience replay, epsilon-greedy |
| `src/snake_env.py` | Environment: state, rewards, pygame rendering |
| `src/train.py` | Main loop: train/watch modes, matplotlib viz |
| `game.py` | Standalone human-playable Snake game |

## Troubleshooting

**ImportError: No module named 'src'**
> Run with: `python -m src.train` (not `python src/train.py`)

**Slow training**
> Set `render=False` in `train.py` line 39

**Model won't load**
> Ensure `main_model.npz` exists. Keys must match: `w1, b1, w2, b2` (lowercase)

## License

Personal project for learning reinforcement learning.
