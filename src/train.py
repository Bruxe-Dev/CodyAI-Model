import numpy as np
import matplotlib.pyplot as plt
from src.snake_env import SnakeEnv
from src.dqn_agent import DQNAgent

def get_curriculum_grid(game_number):
    if game_number <= 1000:
        width, height = 400,300
        phase = 'Foundation'
    
    elif game_number <= 2500:
        width,height = 640,480
        phase = 'Scaling'

    elif game_number <= 4000:
        width,height = 800,600
        phase= 'Mastery'

    else:
        import random
        grid_options = [
            (400, 300),   # Small
            (560, 420),   # Medium-Small
            (640, 480),   # Standard
            (720, 540),   # Medium-Large
            (800, 600),
        ]

        width,height = random.choice(grid_options)
        phase = 'Generalisation'

    return width, height, phase
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
    """
    Train the AI with progressive curriculum learning
    """
    agent = DQNAgent()
    
    num_games = 5000  # Total games to train
    best_score = 0
    
    # Lists to track progress
    all_scores = []
    best_scores = []
    avg_scores = []
    epsilons = []
    grid_history = []  # NEW: Track which grid was used
    
    print("=" * 80)
    print("  PROGRESSIVE CURRICULUM TRAINING")
    print("=" * 80)
    print("\nPhases:")
    print("  1-1000:    Foundation (400x300)   - Learn basics")
    print("  1001-2500: Scaling (640x480)      - Apply skills")  
    print("  2501-4000: Mastery (800x600)      - Handle complexity")
    print("  4001+:     Generalization (mixed) - Adapt to anything")
    print("\n" + "=" * 80)
    print(f"\n{'Game':<8} {'Phase':<16} {'Grid':<12} {'Score':<8} {'Best':<8} {'Avg(50)':<10} {'ε':<8}")
    print("-" * 80)
    
    current_phase = ""  # Track phase changes
    
    for game in range(1, num_games + 1):
        # GET GRID SIZE FROM CURRICULUM
        width, height, phase = get_curriculum_grid(game)
        grid_history.append(f"{width}x{height}")
        
        # ANNOUNCE PHASE CHANGES
        if phase != current_phase:
            current_phase = phase
            print(f"   PHASE: {phase.upper()} - Grid {width}x{height}")
        
        # CREATE ENVIRONMENT WITH CURRICULUM GRID SIZE
        env = SnakeEnv(render=False, width=width, height=height)
        state = env.reset()
        
        # PLAY ONE GAME
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break
        
        # RECORD STATISTICS
        all_scores.append(env.score)
        
        # Save model when new best score achieved
        if env.score > best_score:
            best_score = env.score
            agent.save("main_model.npz")
        
        best_scores.append(best_score)
        avg = float(np.mean(all_scores[-50:]))  # Rolling average of last 50 games
        avg_scores.append(avg)
        epsilons.append(agent.epsilon)
        
        # PRINT PROGRESS every 10 games
        if game % 10 == 0:
            print(f"{game:<8} {phase:<16} {width}x{height:<9} {env.score:<8} "
                  f"{best_score:<8} {avg:<10.2f} {agent.epsilon:<8.4f}")
    
    # TRAINING COMPLETE
    print("\n" + "=" * 80)
    print(f" TRAINING COMPLETE!")
    print(f"  Best Score: {best_score}")
    print(f"  Final Avg:  {avg:.2f}")
    print("=" * 80 + "\n")
    
    # Generate visualization
    plot_results(all_scores, best_scores, avg_scores, epsilons)

def watch():
    env   = SnakeEnv(render=True)
    agent = DQNAgent()
    agent.epsilon = 0.0

    try:
        agent.load("main_model.npz")
    except FileNotFoundError:
        print("No saved model found! Train first: python -m src.train")
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