import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Color definitions
black = (0, 0, 0) # Text color
blue = (24, 124, 245) # Snake color
red = (213, 50, 80) # Food color
green = (27, 127, 27) # Background color
white = (255, 255, 255)

# Game settings
WIDTH = 600
HEIGHT = 400
BLOCK_SIZE = 20
FPS = 400 # Increased speed for training

class SnakeGameAI:
    
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def count_escape_routes(self, pt=None):
        """Count how many immediate directions are safe from a point."""
        if pt is None:
            pt = self.head
        
        directions = [
            Point(pt.x + BLOCK_SIZE, pt.y),  # right
            Point(pt.x - BLOCK_SIZE, pt.y),  # left
            Point(pt.x, pt.y - BLOCK_SIZE),  # up
            Point(pt.x, pt.y + BLOCK_SIZE),  # down
        ]
        
        safe_count = 0
        for d in directions:
            if not self.is_collision(d):
                safe_count += 1
        return safe_count
    
    def flood_fill_count(self, pt=None, max_depth=50):
        """
        Count accessible squares from a point using flood fill.
        This helps detect if the snake is boxing itself into a small area.
        """
        if pt is None:
            pt = self.head
            
        visited = set()
        to_visit = [pt]
        count = 0
        
        while to_visit and count < max_depth:
            current = to_visit.pop(0)
            
            if current in visited:
                continue
            
            # Check bounds
            if current.x < 0 or current.x >= self.w or current.y < 0 or current.y >= self.h:
                continue
            
            # Check if hitting snake body (excluding head for initial position)
            if current in self.snake[1:]:
                continue
                
            visited.add(current)
            count += 1
            
            # Add neighbors
            neighbors = [
                Point(current.x + BLOCK_SIZE, current.y),
                Point(current.x - BLOCK_SIZE, current.y),
                Point(current.x, current.y + BLOCK_SIZE),
                Point(current.x, current.y - BLOCK_SIZE),
            ]
            
            for n in neighbors:
                if n not in visited:
                    to_visit.append(n)
        
        return count
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.history = [] # History of recent head positions
        self.position_visits = {}  # Track how many times each position is visited
        self.direction_history = []  # Track recent directions to detect spinning
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        # Calculate old distance before moving
        old_distance = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # ============================================================
        # SIMPLIFIED REWARD SYSTEM (based on proven implementations)
        # Key: Keep it simple, balanced magnitudes, clear signals
        # ============================================================
        
        reward = 0
        game_over = False
        
        # 3. Check if game over (starvation or collision)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Standard death penalty
            return reward, game_over, self.score
        
        # 4. Food eaten - main positive reward
        if self.head == self.food:
            self.score += 1
            reward = 10  # Standard food reward (balanced with death)
            self._place_food()
            self.frame_iteration = 0
            # Reset tracking on progress
            self.position_visits.clear()
            self.history.clear()
            self.direction_history.clear()
        else:
            self.snake.pop()
            # Distance-based shaping (small values!)
            new_distance = np.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
            if new_distance < old_distance:
                reward = 0.1  # Small reward for moving toward food
            else:
                reward = -0.2  # Slightly stronger penalty for moving away
        
        # 5. Loop tracking (for state awareness, minimal penalty)
        pos_key = (self.head.x, self.head.y)
        self.position_visits[pos_key] = self.position_visits.get(pos_key, 0) + 1
        
        self.history.append(self.head)
        if len(self.history) > 100:
            self.history.pop(0)
        
        self.direction_history.append(self.direction)
        if len(self.direction_history) > 8:
            self.direction_history.pop(0)
        
        # 6. Only penalize DANGEROUS situations (not minor inconveniences)
        escape_routes = self.count_escape_routes(self.head)
        if escape_routes == 0:
            reward -= 5  # Dangerous - no escape
        elif escape_routes == 1:
            reward -= 1  # Risky - only one way out
        
        # 7. Trap detection - only penalize actual traps
        accessible_space = self.flood_fill_count(max_depth=30)
        if accessible_space < len(self.snake) + 2:
            reward -= 3  # Actually trapped

        # 8. update ui and clock
        self._update_ui()
        self.clock.tick(FPS)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def count_escape_routes(self, pt):
        """
        Count how many immediate neighbors are safe to move to.
        Max 4 (Up, Down, Left, Right).
        """
        count = 0
        # Possible neighbors
        neighbors = [
            Point(pt.x + BLOCK_SIZE, pt.y),
            Point(pt.x - BLOCK_SIZE, pt.y),
            Point(pt.x, pt.y + BLOCK_SIZE),
            Point(pt.x, pt.y - BLOCK_SIZE)
        ]
        
        for n in neighbors:
            if not self.is_collision(n):
                count += 1
        return count

    def _update_ui(self):
        self.display.fill(green)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, blue, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

class SnakeGame:
    
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(FPS)
        
        return game_over, self.score
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(green)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, blue, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            break
            
    print('Final Score', score)
        
        
