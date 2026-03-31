import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from snake_env import SnakeEnv
from agent import DQNAgent

# Global stats lists
scores       = []
best_scores  = []
avg_scores   = []
epsilons     = []

def update_graph(frame, ax1, ax2, line_score, line_best, line_avg, line_eps):
    if len(scores) == 0:
        return

    x = list(range(1, len(scores) + 1))

    line_score.set_data(x, scores)
    line_best.set_data(x, best_scores)
    line_avg.set_data(x, avg_scores)
    line_eps.set_data(x, epsilons)

    ax1.set_xlim(1, max(10, len(scores)))
    ax1.set_ylim(0, max(10, max(best_scores) + 5))

    ax2.set_xlim(1, max(10, len(scores)))
    ax2.set_ylim(0, 1.05)


def setup_graph():
    plt.ion()   # interactive mode — updates without blocking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("Snake AI — Live Training Progress", fontsize=14, fontweight="bold")

    # --- Top graph: scores ---
    ax1.set_facecolor("#0f1923")
    ax1.set_xlabel("Game")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.2)

    line_score, = ax1.plot([], [], color="#888888", alpha=0.5,
                            linewidth=1, label="Score per game")
    line_best,  = ax1.plot([], [], color="#2ed573", linewidth=2,
                            label="Best score")
    line_avg,   = ax1.plot([], [], color="#ffa502", linewidth=2,
                            linestyle="--", label="Avg (last 50 games)")
    ax1.legend(loc="upper left")

    # --- Bottom graph: epsilon ---
    ax2.set_facecolor("#0f1923")
    ax2.set_xlabel("Game")
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.grid(True, alpha=0.2)

    line_eps, = ax2.plot([], [], color="#7f77dd", linewidth=2,
                          label="Epsilon")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax1, ax2, line_score, line_best, line_avg, line_eps


def train():
    global scores, best_scores, avg_scores, epsilons

    env   = SnakeEnv(render=False)
    agent = DQNAgent()

    num_games  = 1000
    best_score = 0

    fig, ax1, ax2, line_score, line_best, line_avg, line_eps = setup_graph()

    print("Training started! Watch the graph update live.\n")
    print(f"{'Game':<8} {'Score':<8} {'Best':<8} {'Avg(50)':<10} {'Epsilon'}")
    print("-" * 55)

    for game in range(1, num_games + 1):
        state = env.reset()

        while True:
            action             = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break

        # --- Record stats ---
        scores.append(env.score)

        if env.score > best_score:
            best_score = env.score
            agent.save("snake_brain.npz")

        best_scores.append(best_score)
        avg = float(np.mean(scores[-50:]))
        avg_scores.append(avg)
        epsilons.append(agent.epsilon)

        # --- Update graph every 5 games ---
        if game % 5 == 0:
            update_graph(None, ax1, ax2,
                         line_score, line_best, line_avg, line_eps)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # --- Print to terminal every 10 games ---
        if game % 10 == 0:
            print(f"{game:<8} {env.score:<8} {best_score:<8} {avg:<10.2f} {agent.epsilon:.4f}")

    print(f"\nTraining complete! Best score: {best_score}")

    # Save final graph as image
    plt.savefig("training_progress.png", dpi=150, bbox_inches="tight")
    print("Graph saved to training_progress.png")

    plt.ioff()
    plt.show()   # keep graph open after training ends


def watch():
    env   = SnakeEnv(render=True)
    agent = DQNAgent()
    agent.epsilon = 0.0

    try:
        agent.load("snake_brain.npz")
    except FileNotFoundError:
        print("No saved model found! Train first: python train.py")
        return

    print("Watching AI play...\n")
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