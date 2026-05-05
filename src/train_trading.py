import numpy as np
import matplotlib.pyplot as plt
from src.trading_env import TradingEnv, download_stock_data
from src.trading_agent import TradingDQNAgent

def plot_trading_results(returns, portfolio_values, epsilons):
    """
    Visualize trading performance
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Trading AI — Training Results", fontsize=14, fontweight="bold")
    
    episodes = list(range(1, len(returns) + 1))
    
    # Returns per episode
    ax1.set_facecolor("#0f1923")
    ax1.plot(episodes, returns, color="#888888", alpha=0.6, linewidth=1, label="Episode Return %")
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    
    # Moving average
    if len(returns) >= 10:
        moving_avg = np.convolve(returns, np.ones(10)/10, mode='valid')
        ax1.plot(range(10, len(returns)+1), moving_avg, color="#ffa502", linewidth=2, label="10-Episode Avg")
    
    ax1.set_ylabel("Return %")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)
    
    # Portfolio value
    ax2.set_facecolor("#0f1923")
    ax2.plot(episodes, portfolio_values, color="#2ed573", linewidth=2, label="Final Portfolio Value")
    ax2.axhline(y=10000, color='white', linestyle='--', alpha=0.3, label="Initial Balance")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)
    
    # Epsilon
    ax3.set_facecolor("#0f1923")
    ax3.plot(episodes, epsilons, color="#7f77dd", linewidth=2, label="Epsilon (Exploration)")
    ax3.set_ylabel("Epsilon")
    ax3.set_xlabel("Episode")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("trading_progress.png", dpi=150, bbox_inches="tight")
    print("✓ Graph saved → trading_progress.png")
    plt.show()

def train_trading_agent(ticker='AAPL', episodes=500):
    """
    Train trading AI
    
    Args:
        ticker: Stock symbol to trade
        episodes: Number of training episodes
    """
    print("=" * 80)
    print(f"  🤖 TRAINING TRADING AI ON {ticker}")
    print("=" * 80)
    
    # Download stock data
    print("\n📊 Downloading market data...")
    stock_data = download_stock_data(ticker, '2018-01-01', '2023-12-31')
    
    # Create environment and agent
    env = TradingEnv(stock_data, initial_balance=10000)
    agent = TradingDQNAgent()
    
    # Training metrics
    all_returns = []
    all_portfolio_values = []
    all_epsilons = []
    best_return = -float('inf')
    
    print(f"\n🎯 Starting training for {episodes} episodes...\n")
    print(f"{'Episode':<10} {'Return %':<12} {'Portfolio':<15} {'Trades':<10} {'Epsilon':<10}")
    print("-" * 80)
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        # Trade for one episode
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        
        # Record results
        episode_return = env.get_return()
        portfolio_value = env.get_portfolio_value()
        
        all_returns.append(episode_return)
        all_portfolio_values.append(portfolio_value)
        all_epsilons.append(agent.epsilon)
        
        # Save best model
        if episode_return > best_return:
            best_return = episode_return
            agent.save(f"best_trading_model_{ticker}.npz")
        
        # Print progress
        if episode % 10 == 0:
            print(f"{episode:<10} {episode_return:>+10.2f}% ${portfolio_value:>12.2f} "
                  f"{env.total_trades:<10} {agent.epsilon:<10.4f}")
    
    print("\n" + "=" * 80)
    print("   TRAINING COMPLETE!")
    print(f"  Best Return: {best_return:+.2f}%")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    print("=" * 80 + "\n")
    
    # Plot results
    plot_trading_results(all_returns, all_portfolio_values, all_epsilons)
    
    return agent

if __name__ == "__main__":
    # Train on Apple stock
    agent = train_trading_agent(ticker='AAPL', episodes=500)