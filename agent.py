import random
import numpy as np
from collections import deque
from snake import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(14, 256, 3)  # 14 features, smaller network for faster learning
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Anti-loop tracking
        self.recent_actions = deque(maxlen=12)  # Track last 12 actions
    
    def reset_tracking(self):
        """Reset loop tracking at start of new game."""
        self.recent_actions.clear()

    def get_state(self, game):
        """
        Simplified state representation based on proven implementations.
        14 features total - enough info without overwhelming the network.
        """
        head = game.snake[0]
        
        # Points around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Get relative direction points (straight, right, left from current direction)
        if dir_r:
            straight_pt, right_pt, left_pt = point_r, point_d, point_u
        elif dir_l:
            straight_pt, right_pt, left_pt = point_l, point_u, point_d
        elif dir_u:
            straight_pt, right_pt, left_pt = point_u, point_r, point_l
        else:  # dir_d
            straight_pt, right_pt, left_pt = point_d, point_l, point_r

        state = [
            # Danger in each relative direction (3 features) - CORE
            game.is_collision(straight_pt),
            game.is_collision(right_pt),
            game.is_collision(left_pt),
            
            # Current direction (4 features) - CORE
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food direction (4 features) - CORE
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
            
            # Escape routes after each move (3 features) - helps avoid traps
            game.count_escape_routes(straight_pt) / 4.0 if not game.is_collision(straight_pt) else 0,
            game.count_escape_routes(right_pt) / 4.0 if not game.is_collision(right_pt) else 0,
            game.count_escape_routes(left_pt) / 4.0 if not game.is_collision(left_pt) else 0,
        ]
        
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        # Learning Rate Decay
        if self.n_games > 500:
            self.trainer.optimizer.learning_rate = 0.0001
        elif self.n_games > 100:
            self.trainer.optimizer.learning_rate = 0.0005
        else:
            self.trainer.optimizer.learning_rate = 0.001

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game=None):
        """
        Get action with safety overrides to prevent death and loops.
        """
        # Extract danger info from state (first 3 features)
        danger_straight = state[0] > 0.5
        danger_right = state[1] > 0.5
        danger_left = state[2] > 0.5
        
        # Extract escape routes from state (last 3 features before any extras)
        escape_straight = state[11] if len(state) > 11 else 0
        escape_right = state[12] if len(state) > 12 else 0
        escape_left = state[13] if len(state) > 13 else 0
        
        # Determine which moves are safe
        safe_moves = []
        if not danger_straight:
            safe_moves.append((0, escape_straight))  # (move_idx, escape_routes)
        if not danger_right:
            safe_moves.append((1, escape_right))
        if not danger_left:
            safe_moves.append((2, escape_left))
        
        # Epsilon-greedy exploration
        self.epsilon = 120 - self.n_games 
        if self.epsilon < 2:
            self.epsilon = 2
        
        # Get model's preferred action
        if random.randint(0, 200) < self.epsilon:
            # Random exploration - but only pick from SAFE moves if possible
            if safe_moves:
                move = random.choice([m[0] for m in safe_moves])
            else:
                move = random.randint(0, 2)  # All dangerous, pick random
        else:
            state0 = state.reshape(1, -1)
            prediction = self.model.predict(state0)
            move = np.argmax(prediction)
        
        # ========== SAFETY OVERRIDE 1: Prevent immediate death ==========
        chosen_is_safe = (move == 0 and not danger_straight) or \
                         (move == 1 and not danger_right) or \
                         (move == 2 and not danger_left)
        
        if not chosen_is_safe and safe_moves:
            # Model chose death - override with best safe move
            # Prefer move with most escape routes
            safe_moves.sort(key=lambda x: x[1], reverse=True)
            move = safe_moves[0][0]
        
        # ========== SAFETY OVERRIDE 2: Anti-loop detection ==========
        if len(self.recent_actions) >= 8:
            # Check for repeating pattern (e.g., right-left-right-left)
            recent = list(self.recent_actions)[-8:]
            
            # Pattern 1: Alternating (0,1,0,1 or 1,2,1,2 etc)
            is_alternating = all(recent[i] == recent[i+2] for i in range(6))
            
            # Pattern 2: Same action repeatedly  
            is_stuck = len(set(recent[-4:])) == 1
            
            # Pattern 3: Circular (0,1,2,0,1,2...)
            is_circular = recent[-3:] == recent[-6:-3] if len(recent) >= 6 else False
            
            if is_alternating or is_stuck or is_circular:
                # Force a different safe action
                other_safe = [m[0] for m in safe_moves if m[0] != move]
                if other_safe:
                    # Pick the one with best escape routes
                    best_alt = max(other_safe, key=lambda m: 
                        next((s[1] for s in safe_moves if s[0] == m), 0))
                    move = best_alt
        
        # ========== SAFETY OVERRIDE 3: Prefer more escape routes ==========
        # If current move has 0-1 escape routes but another safe move has more
        if safe_moves and len(safe_moves) > 1:
            current_escapes = next((s[1] for s in safe_moves if s[0] == move), 0)
            best_escapes = max(s[1] for s in safe_moves)
            
            # If there's a significantly better option, take it
            if best_escapes > current_escapes + 0.25:  # At least 1 more escape route
                move = max(safe_moves, key=lambda x: x[1])[0]
        
        # Track this action
        self.recent_actions.append(move)
        
        # Build final move
        final_move = [0, 0, 0]
        final_move[move] = 1
        
        return final_move

def train():
    agent = Agent()
    game = SnakeGameAI()
    
    # Try loading existing model
    record = 0
    try:
        loaded_model, extra_data = Linear_QNet.load()
        if loaded_model:
             agent.model = loaded_model
             agent.trainer.model = loaded_model # Update trainer reference
             if 'n_games' in extra_data:
                 agent.n_games = extra_data['n_games']
             if 'record' in extra_data:
                 record = extra_data['record']
             print(f"Model loaded. Resuming from Game {agent.n_games}, Record {record}")
    except Exception as e:
        print(f"No model found or error loading ({e}), starting fresh.")
        
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    
    try:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old, game)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.reset_tracking()  # Reset loop detection for new game
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(extra_data={'n_games': agent.n_games, 'record': record})

                print('Game', agent.n_games, 'Score', score, 'Record', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving model...")
        agent.model.save(extra_data={'n_games': agent.n_games, 'record': record})
        print("Model saved successfully. Exiting.")


if __name__ == '__main__':
    train()
