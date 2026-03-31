import numpy as np
from snake_env import SnakeEnv
from agent import DQNAgent

def train():
    env   = SnakeEnv(render=True)  
    agent = DQNAgent()

    num_games    = 1000
    best_score   = 0
    total_scores = []

    print("Starting training...\n")
    print(f"{'Game':<8} {'Score':<8} {'Best':<8} {'Epsilon':<10} {'Memory'}")
    print("-" * 50)

    for game in range(1, num_games + 1):
        state = env.reset()
        total_reward = 0

        while True:
            # 1. Agent picks an action
            action = agent.act(state)

            # 2. Game runs that action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 3. Agent remembers what happened
            agent.remember(state, action, reward, next_state, done)

            # 4. Agent learns from past memories
            agent.learn()

            state = next_state

            if done:
                break

        # --- End of game bookkeeping ---
        total_scores.append(env.score)

        if env.score > best_score:
            best_score = env.score
            agent.save("snake_brain.npz")   # save when we beat the record!

        # Print progress every 10 games
        if game % 10 == 0:
            avg = np.mean(total_scores[-50:])   # avg of last 50 games
            print(f"{game:<8} {env.score:<8} {best_score:<8} "
                  f"{agent.epsilon:<10.3f} {len(agent.memory)}")

    print("\nTraining complete!")
    print(f"Best score achieved: {best_score}")


def watch():
    env   = SnakeEnv(render=True)
    agent = DQNAgent()
    agent.epsilon = 0.0        # no random moves when watching

    try:
        agent.load("snake_brain.npz")
    except FileNotFoundError:
        print("No saved model found! Train first with: python train.py train")
        return

    print("Watching AI play... (close window to stop)\n")

    while True:
        state = env.reset()
        while True:
            action = agent.act(state)
            state, _, done = env.step(action)
            if done:
                print(f"Score: {env.score}")
                break


# ================================================================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch()
    else:
        train()