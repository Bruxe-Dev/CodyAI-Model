import pygame
import numpy as np
from src_trade.trading_env import ForexEnv, download_forex_data, ForexAction
from src_trade.trading_agent import TradingDQNAgent

# Pygame setup
pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("🤖 Cody AI - Live Forex Trading")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (15, 25, 35)
TEXT_COLOR = (255, 255, 255)
GREEN = (46, 213, 115)
RED = (252, 92, 101)
BLUE = (52, 152, 219)
ORANGE = (255, 165, 0)

# Fonts
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 24)

def draw_text(text, x, y, font, color=TEXT_COLOR):
    """Draw text on screen"""
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def draw_chart(price_history, x, y, width, height):
    """Draw price chart"""
    if len(price_history) < 2:
        return
    
    # Normalize prices to fit in chart
    prices = np.array(price_history[-100:])  # Last 100 prices
    min_price = prices.min()
    max_price = prices.max()
    price_range = max_price - min_price
    
    if price_range == 0:
        return
    
    # Draw chart background
    pygame.draw.rect(screen, (20, 30, 40), (x, y, width, height))
    pygame.draw.rect(screen, (50, 60, 70), (x, y, width, height), 2)
    
    # Draw price line
    points = []
    for i, price in enumerate(prices):
        px = x + (i / len(prices)) * width
        py = y + height - ((price - min_price) / price_range) * height
        points.append((px, py))
    
    if len(points) > 1:
        pygame.draw.lines(screen, BLUE, False, points, 2)

def visualize_trading(agent_path, forex_pair='EURUSD=X'):
    """
    Live visualization of AI trading
    """
    print(f"🎮 Loading AI and starting live visualization...")
    
    # Load Forex data
    # FIX: Only pass the forex_pair, as the date range is handled automatically now
    forex_data = download_forex_data(forex_pair)
    env = ForexEnv(forex_data, initial_balance=10000, leverage=10)
    
    # Load trained agent
    agent = TradingDQNAgent(input_size=18) # Added input_size to ensure it matches
    try:
        agent.load(agent_path)
        agent.epsilon = 0  # No exploration
        print("✓ Model loaded successfully")
    except:
        print("⚠️  No saved model found, using untrained agent")
    
    # Trading state
    state = env.reset()
    done = False
    price_history = []
    portfolio_history = []
    
    running = True
    paused = False
    speed = 1  # Steps per frame
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(speed + 1, 10)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed - 1, 1)
                elif event.key == pygame.K_r:
                    # Reset
                    state = env.reset()
                    done = False
                    price_history = []
                    portfolio_history = []
        
        # Clear screen
        screen.fill(BG_COLOR)
        
        # Update trading (if not paused and not done)
        if not paused and not done:
            for _ in range(speed):
                # Get AI action
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                state = next_state
                
                # Record history
                current_price = env.forex_data.iloc[env.current_step]['Close']
                price_history.append(current_price)
                portfolio_history.append(env.get_portfolio_value())
                
                if done:
                    break
        
        # Draw UI
        # Title
        draw_text("🤖 CODY AI - LIVE FOREX TRADING", 50, 30, font_large, ORANGE)
        
        # Trading pair and status
        draw_text(f"Pair: {forex_pair}", 50, 90, font_medium)
        status_color = GREEN if not paused else RED
        status_text = "TRADING" if not paused else "PAUSED"
        draw_text(f"Status: {status_text}", 300, 90, font_medium, status_color)
        
        # Portfolio info
        portfolio_value = env.get_portfolio_value()
        portfolio_return = env.get_return()
        return_color = GREEN if portfolio_return >= 0 else RED
        
        draw_text(f"Balance: ${env.balance:,.2f}", 50, 140, font_medium)
        draw_text(f"Portfolio: ${portfolio_value:,.2f}", 300, 140, font_medium)
        draw_text(f"Return: {portfolio_return:+.2f}%", 600, 140, font_medium, return_color)
        
        # Position info
        position_text = "NONE"
        position_color = TEXT_COLOR
        if env.position == 1:
            position_text = "LONG (BUY)"
            position_color = GREEN
        elif env.position == -1:
            position_text = "SHORT (SELL)"
            position_color = RED
        
        draw_text(f"Position: {position_text}", 50, 190, font_medium, position_color)
        draw_text(f"Trades: {env.total_trades}", 400, 190, font_medium)
        draw_text(f"Step: {env.current_step}/{env.max_steps}", 600, 190, font_medium)
        
        # Price chart
        draw_text("Price Chart (Last 100 Steps)", 50, 250, font_small)
        draw_chart(price_history, 50, 280, 500, 200)
        
        # Portfolio chart
        draw_text("Portfolio Value", 600, 250, font_small)
        draw_chart(portfolio_history, 600, 280, 500, 200)
        
        # Current action (if trading)
        if not paused and not done:
            current_action = agent.act(state)
            action_text = ForexAction(current_action).name
            action_color = GREEN if current_action == 1 else (RED if current_action == 2 else TEXT_COLOR)
            draw_text(f"AI Decision: {action_text}", 50, 520, font_large, action_color)
        
        # Controls
        draw_text("Controls:", 50, 600, font_medium)
        draw_text("SPACE - Pause/Resume", 50, 640, font_small)
        draw_text("↑/↓ - Speed", 50, 670, font_small)
        draw_text("R - Reset", 50, 700, font_small)
        draw_text(f"Speed: {speed}x", 50, 730, font_small, ORANGE)
        
        # Episode complete message
        if done:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # Results
            final_return = env.get_return()
            result_color = GREEN if final_return >= 0 else RED
            
            draw_text("EPISODE COMPLETE!", SCREEN_WIDTH // 2 - 200, 300, font_large, ORANGE)
            draw_text(f"Final Return: {final_return:+.2f}%", SCREEN_WIDTH // 2 - 150, 380, font_large, result_color)
            draw_text(f"Total Trades: {env.total_trades}", SCREEN_WIDTH // 2 - 120, 450, font_medium)
            draw_text("Press R to restart", SCREEN_WIDTH // 2 - 120, 520, font_medium)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    pygame.quit()


if __name__ == "__main__":
    model_path = "best_EURUSD=X.npz" 
    visualize_trading(model_path, "EURUSD=X")