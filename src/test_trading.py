from src.trading_env import TradingEnv, download_stock_data
from src.trading_agent import TradingDQNAgent
import matplotlib.pyplot as plt

def test_trading_agent(ticker='AAPL'):
    """
    Test trained model on unseen data
    """
    print(f"\n🧪 Testing model on {ticker}...")
    
    # Download NEW data (2024 - unseen by model)
    test_data = download_stock_data(ticker, '2024-01-01', '2024-12-31')
    
    # Create environment
    env = TradingEnv(test_data, initial_balance=10000)
    
    # Load trained agent
    agent = TradingDQNAgent()
    agent.load(f"best_trading_model_{ticker}.npz")
    agent.epsilon = 0  # No exploration, only exploitation
    
    # Test
    state = env.reset()
    done = False
    actions_taken = []
    
    while not done:
        action = agent.act(state)
        actions_taken.append(action)
        next_state, reward, done = env.step(action)
        state = next_state
    
    # Results
    final_return = env.get_return()
    final_value = env.get_portfolio_value()
    
    print(f"\n📈Test Results:")
    print(f"  Initial Balance: ${env.initial_balance}")
    print(f"  Final Portfolio: ${final_value:.2f}")
    print(f"  Return: {final_return:+.2f}%")
    print(f"  Total Trades: {env.total_trades}")
    
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(env.net_worth_history, linewidth=2, color='#2ed573')
    plt.axhline(y=10000, color='white', linestyle='--', alpha=0.5, label='Initial Balance')
    plt.title(f"{ticker} - AI Trading Performance (Test Data)")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"test_results_{ticker}.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    test_trading_agent('AAPL')