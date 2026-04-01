import numpy as np
import matplotlib.pyplot as plt
from src.snake_env import SnakeEnv
from src.dqn_agent import DQNAgent

def plot_results(scores, best_scores, avg_scores, epsilons):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Cody AI — Training Results", fontsize=14, fontweight="bold")

    games = list(range(1, len(scores) + 1))

    # --- Top: scores ---
    ax1.set_facecolor("#0f1923")
    ax1.plot(games, scores,      color="#888888", alpha=0.5,
             linewidth=1,  label="Score per game")
    ax1.plot(games, best_scores, color="#2ed573",
             linewidth=2,  label="Best score")
    ax1.plot(games, avg_scores,  color="#ffa502",
             linewidth=2,  linestyle="--", label="Avg last 50 games")
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Bottom: epsilon ---
    ax2.set_facecolor("#0f1923")
    ax2.plot(games, epsilons, color="#7f77dd", linewidth=2, label="Epsilon")
    ax2.set_ylabel("Epsilon")
    ax2.set_xlabel("Game")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150, bbox_inches="tight")
    print("Graph saved → training_progress.png")
    plt.show()


def train():
    env   = SnakeEnv(render=True)   # set render=False for faster training (no window)
    agent = DQNAgent()

    num_games  = 1000
    best_score = 0

    all_scores  = []
    best_scores = []
    avg_scores  = []
    epsilons    = []

    print("Training started! Graph will show when done.\n")
    print(f"{'Game':<8} {'Score':<8} {'Best':<8} {'Avg(50)':<10} {'Epsilon'}")
    print("-" * 55)

    for game in range(1, num_games + 1):
        state = env.reset()

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break

        # --- Record stats ---
        all_scores.append(env.score)

        if env.score > best_score:
            best_score = env.score
            agent.save("main_model.npz")

        best_scores.append(best_score)
        avg = float(np.mean(all_scores[-50:]))
        avg_scores.append(avg)
        epsilons.append(agent.epsilon)

        if game % 10 == 0:
            print(f"{game:<8} {env.score:<8} {best_score:<8} "
                  f"{avg:<10.2f} {agent.epsilon:.4f}")

    print(f"\nTraining complete! Best score: {best_score}")
    plot_results(all_scores, best_scores, avg_scores, epsilons)


def watch():
    """Load trained model and watch it play."""
    env   = SnakeEnv(render=True)
    agent = DQNAgent()
    agent.epsilon = 0.0

    try:
        agent.load("main_model.npz")
    except FileNotFoundError:
        print("No saved model found! Train first: python train.py")
        return

    print("Watching AI play... (close window to stop)\n")
    game = 0

    while True:
        state = env.reset()
        game += 1
        while True:
            action = agent.act(state)
            state, _, done = env.step(action)
            if done:
                print(f"Game {game} | Score: {env.score}")
                break


# ================================================================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "watch":
        watch()
    else:
        train()