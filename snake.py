import pygame
import random
import numpy as np

# Color definitions
black = (0, 0, 0) # Text color
blue = (24, 124, 245) # Snake color
red = (213, 50, 80) # Food color
green = (27, 127, 27) # Background color

# Game settings
WIDTH = 600
HEIGHT = 400
BLOCK_SIZE = 10
FPS = 15 # Controls the speed of the snake

# Get current state of the game for the RL agent
def get_state(self):
    head_x, head_y = self.snake[0]
    dir_x, dir_y = self.direction

    # Danger
    danger_straight = self.is_collision((head_x + dir_x * BLOCK_SIZE, head_y + dir_y * BLOCK_SIZE))
    danger_right = self.is_collision((head_x + dir_y * BLOCK_SIZE, head_y - dir_x * BLOCK_SIZE))
    danger_left = self.is_collision((head_x - dir_y * BLOCK_SIZE, head_y + dir_x * BLOCK_SIZE))

    # Food location
    food_x, food_y, = self.food

    food_left =  food_x < head_x
    food_right = food_x > head_x
    food_up = food_y < head_y
    food_down = food_y > head_y

    state = [
        danger_left, danger_straight, danger_right,
        dir_x == 1, dir_x == -1, dir_y == 1, dir_y == -1,
        food_left, food_right, food_up, food_down
    ]
    return np.array(state, dtype=int)

# Check if the given point collides with walls or the snake itself
def is_collision(self, point):
    head_x, head_y = point
    # Check for wall collisions
    if head_x < 0 or head_x >= WIDTH or head_y < 0 or head_y >= HEIGHT:
        return True
    # Check for self collisions
    if point in self.snake[1:]:
        return True
    return False

# Apply the action to change the snake's direction
def apply_acction(self, action):
    dir_x, dir_y = self.direction

    if action == 0: # Straight
        return
    elif action == 1: # Right turn
        self.direction = (dir_y, -dir_x)
    elif action == 2: # Left turn
        self.direction = (-dir_y, dir_x)

# Take a step in the game based on the action
def step(self, action):
    self.apply_acction(action)
    self.move_snake()

    # Reward
    reward = 0
    if self.game_over:
        reward = -10
    elif self.snake[0] == self.food:
        reward = 10
    
    next_state = self.get_state()
    return next_state, reward, self.game_over, {}


class SnakeGame:
    # Initialize pygame
    pygame.init()
    # Set up the game window
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake!')
        self.clock = pygame.time.Clock()
        self.reset_game()
    # Starting game state
    def reset_game(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)] # starts snake in center of the screen
        self.direction = (1, 0) # makes snake go to the right
        self.food = self.spawn_food() # first food spawn
        self.score = 0 # initial score
        self.game_over = False
    # Spawn food at random location not occupied by the snake 
    def spawn_food(self):
        while True:
            # Get random position
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            # As long as food is not on the snake, return position
            if (x, y) not in self.snake:
                return (x, y)
    # Handle user input for snake direction
    def handle_keys(self):
        for event in pygame.event.get():
            # If user closes window, quit the game
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.KEYDOWN:
                # if left key pressed and not going left, go left
                if event.key == pygame.K_LEFT and self.direction != (1, 0):
                    self.direction = (-1, 0)
                # if right key pressed and not going right, go right
                elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                    self.direction = (1, 0)
                # if up key pressed and not going up, go up
                elif event.key == pygame.K_UP and self.direction != (0, 1):
                    self.direction = (0, -1)
                # if down key pressed and not going down, go down
                elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                    self.direction = (0, 1)
    # Move the snake in the current direction
    def move_snake(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        # moves snake head by block size in the current direction
        new_head = (head_x + dir_x * BLOCK_SIZE, head_y + dir_y * BLOCK_SIZE)

        # Check for collisions with walls or self, ends game if collision occurs
        if (new_head[0] < 0 or new_head[0] >= WIDTH or # hits left or right wall
            new_head[1] < 0 or new_head[1] >= HEIGHT or # hits top or bottom wall
            new_head in self.snake): # hits itself
            self.game_over = True
            return
        # Add new head to the snake
        self.snake.insert(0, new_head)
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        # If food not eaten, remove tail segment
        else:
            self.snake.pop()
    # Draw the game state on the screen
    def draw(self):
        # create background
        self.screen.fill(green)
        # draw each block of snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, blue, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        # draw food
        pygame.draw.rect(self.screen, red, (*self.food, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.SysFont(None, 35)
        # draw score at top left corner
        score_text = font.render(f'Score: {self.score}', True, black)
        self.screen.blit(score_text, [0, 0])
        # if game over, display game over message
        if self.game_over:
            over_text = font.render('Game Over!', True, red)
            middle_text = font.render('Press any arrow key to play again', True, black)
            self.screen.blit(over_text, [WIDTH // 2 - 80, HEIGHT // 3])
            self.screen.blit(middle_text, [WIDTH // 6, HEIGHT // 3 + 40])
        # refresh display
        pygame.display.flip()
    # Main game loop
    def run(self):
        # Loop until the game is over
        while not self.game_over:
            self.handle_keys()
            self.move_snake()
            self.draw()
            self.clock.tick(FPS)
        # After game over, wait for user input to restart or quit
        while True:
            for event in pygame.event.get():
                # if user closes window, quit the game
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    # if user presses arrow key, restart game
                    if event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        self.reset_game()
                        self.run()
                        return
                    # if user presses backspace, quit the game
                    elif event.key == pygame.K_BACKSPACE:
                        pygame.quit()
                        return
if __name__ == "__main__":
    game = SnakeGame()
    game.run()