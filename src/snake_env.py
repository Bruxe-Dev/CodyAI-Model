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
        head = self.snake[0]
        dir  = self.direction

        # Points one step ahead in each relative direction
        point_straight = self._next_point(dir)
        point_right    = self._next_point(self._turn_right(dir))
        point_left     = self._next_point(self._turn_left(dir))

        state = [
            # --- 3 danger sensors ---
            self._is_dangerous(point_straight),   # wall or body ahead?
            self._is_dangerous(point_right),
            self._is_dangerous(point_left),

            # --- 4 current direction (one-hot) ---
            dir == Direction.UP,
            dir == Direction.DOWN,
            dir == Direction.LEFT,
            dir == Direction.RIGHT,

            # --- 4 food direction ---
            self.food[1] < head[1],   # food is up
            self.food[1] > head[1],   # food is down
            self.food[0] < head[0],   # food is left
            self.food[0] > head[0],   # food is right
        ]

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

        if new_head == self.food:
            reward = 10
            self.score += 1
            self.steps = 0              # reset loop counter on food
            self.max_steps = len(self.snake) * 100
            self._place_food()
        else:
            reward = 1                  # small reward for surviving
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