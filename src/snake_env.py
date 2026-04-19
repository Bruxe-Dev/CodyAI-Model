import pygame
import random
import numpy as np
import sys
from enum import Enum

pygame.init()

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SnakeEnv:
    def __init__(self, render=True):
        """
        render=False → runs invisibly (fast training, no window)
        render=True  → shows the game window (watch it play)
        """
        self.render_mode = render
        self.w = SCREEN_WIDTH
        self.h = SCREEN_HEIGHT

        if self.render_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        """Start a fresh game. Returns the initial state."""
        self.direction = Direction.RIGHT
        head_x = (self.w // 2 // BLOCK_SIZE) * BLOCK_SIZE
        head_y = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE

        self.snake = [
            [head_x, head_y],
            [head_x - BLOCK_SIZE, head_y],
            [head_x - 2 * BLOCK_SIZE, head_y],
        ]

        self.score = 0
        self.steps = 0                      # track steps to detect loops
        self.max_steps = len(self.snake) * 100  # reset if stuck in loop
        self._place_food()
        return self.get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = [x, y]
            if self.food not in self.snake:
                break
    def get_state(self):
        """
        Enhanced state with 23 features
        """
        head = self.snake[0]
        
        # Current direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Points around head (1 block away)
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        # Points 2 blocks away
        point_l2 = [head[0] - 2*BLOCK_SIZE, head[1]]
        point_r2 = [head[0] + 2*BLOCK_SIZE, head[1]]
        point_u2 = [head[0], head[1] - 2*BLOCK_SIZE]
        point_d2 = [head[0], head[1] + 2*BLOCK_SIZE]
        
        # Danger 1 block away (straight)
        danger_straight = (
            (dir_r and self._is_dangerous(point_r)) or 
            (dir_l and self._is_dangerous(point_l)) or 
            (dir_u and self._is_dangerous(point_u)) or 
            (dir_d and self._is_dangerous(point_d))
        )
        
        # Danger 1 block away (right)
        danger_right = (
            (dir_u and self._is_dangerous(point_r)) or 
            (dir_d and self._is_dangerous(point_l)) or 
            (dir_l and self._is_dangerous(point_u)) or 
            (dir_r and self._is_dangerous(point_d))
        )
        
        # Danger 1 block away (left)
        danger_left = (
            (dir_d and self._is_dangerous(point_r)) or 
            (dir_u and self._is_dangerous(point_l)) or 
            (dir_r and self._is_dangerous(point_u)) or 
            (dir_l and self._is_dangerous(point_d))
        )
        
        # Danger 2 blocks away
        danger_ahead_2 = (
            (dir_r and self._is_dangerous(point_r2)) or 
            (dir_l and self._is_dangerous(point_l2)) or 
            (dir_u and self._is_dangerous(point_u2)) or 
            (dir_d and self._is_dangerous(point_d2))
        )
        
        # Food location (binary)
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        # Distance to food (normalized 0-1)
        food_distance_x = abs(self.food[0] - head[0]) / self.w
        food_distance_y = abs(self.food[1] - head[1]) / self.h
        
        # Snake length (normalized)
        snake_length = len(self.snake) / 100.0
        
        # Available space in each direction (normalized)
        space_left = head[0] / self.w
        space_right = (self.w - head[0]) / self.w
        space_up = head[1] / self.h
        space_down = (self.h - head[1]) / self.h
        
        # Build state array (23 features total)
        state = [
            # Danger (4 features)
            int(danger_straight),      # 0
            int(danger_right),         # 1
            int(danger_left),          # 2
            int(danger_ahead_2),       # 3
            
            # Direction (4 features)
            int(dir_l),                # 4
            int(dir_r),                # 5
            int(dir_u),                # 6
            int(dir_d),                # 7
            
            # Food direction (4 features)
            int(food_left),            # 8
            int(food_right),           # 9
            int(food_up),              # 10
            int(food_down),            # 11
            
            # Food distance (2 features)
            food_distance_x,           # 12
            food_distance_y,           # 13
            
            # Snake length (1 feature)
            snake_length,              # 14
            
            # Available space (4 features)
            space_left,                # 15
            space_right,               # 16
            space_up,                  # 17
            space_down,                # 18
        ]
        
        # VERIFY: Should be 23 features
        assert len(state) == 23, f"State has {len(state)} features, expected 23"
        
        return np.array(state, dtype=float)

    def _next_point(self, direction):
        head = self.snake[0]
        if direction == Direction.UP:    return [head[0], head[1] - BLOCK_SIZE]
        if direction == Direction.DOWN:  return [head[0], head[1] + BLOCK_SIZE]
        if direction == Direction.LEFT:  return [head[0] - BLOCK_SIZE, head[1]]
        if direction == Direction.RIGHT: return [head[0] + BLOCK_SIZE, head[1]]

    def _is_dangerous(self, point):
        """True if this point is a wall or snake body."""
        x, y = point
        wall = x < 0 or x >= self.w or y < 0 or y >= self.h
        body = point in self.snake[1:]
        return float(wall or body)

    def _turn_right(self, d):
        order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        return order[(order.index(d) + 1) % 4]

    def _turn_left(self, d):
        order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        return order[(order.index(d) - 1) % 4]

    def step(self, action):
        """
        action: 0=turn left, 1=go straight, 2=turn right
        returns: (next_state, reward, done)
        """
        self.steps += 1

        # Handle pygame quit even in training mode
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # Convert relative action to absolute direction
        if action == 0:    self.direction = self._turn_left(self.direction)
        elif action == 2:  self.direction = self._turn_right(self.direction)
        # action == 1 → keep going straight

        # Move snake
        new_head = self._next_point(self.direction)
        self.snake.insert(0, new_head)

        # --- Reward logic ---
        reward = 0
        done   = False

        if self._is_dangerous(new_head):
            reward = -10
            done   = True
            self.snake.pop()
            if self.render_mode:
                self._draw()
            return self.get_state(), reward, done

        # Calculate distance to food before and after move
        old_distance = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_head == self.food:
            reward = 10  # Big reward for eating
            self.score += 1
            self.steps = 0
            self.max_steps = len(self.snake) * 100
            self._place_food()
        else:
            # Reward for moving closer, penalty for moving away
            if new_distance < old_distance:
                reward = 1  # Moving toward food
            else:
                reward = -1  # Moving away from food
            self.snake.pop()

        # Punish the AI if it loops forever without eating
        if self.steps >= self.max_steps:
            reward = -10
            done   = True

        if self.render_mode:
            self._draw()

        return self.get_state(), reward, done


    def _draw(self):
        self.display.fill((15, 25, 35))

        # Draw snake
        for i, seg in enumerate(self.snake):
            color = (46, 213, 115) if i == 0 else (34, 166, 90)
            pygame.draw.rect(self.display, color,
                             pygame.Rect(seg[0], seg[1], BLOCK_SIZE, BLOCK_SIZE),
                             border_radius=4)

        # Draw food
        pygame.draw.rect(self.display, (252, 92, 101),
                         pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE),
                         border_radius=BLOCK_SIZE // 2)

        # Score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)