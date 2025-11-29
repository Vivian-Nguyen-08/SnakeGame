import snake          # or from snake import Game if game class exists
from model import QAgent  # replace with actual agent class name
import json

# Initialize the game
game = snake.Game()      # or however Game is initialized
agent = QAgent()         # load your trained model if available

data_log = []

for episode in range(10):  # run 10 test games
    state = game.reset()
    done = False
    episode_data = []

    while not done:
        # Get action from the RL agent
        action = agent.get_action(state)

        # Step the game
        next_state, reward, done, info = game.step(action)

        # Save state-action-reward
        episode_data.append({
            'state': state.tolist(),   # convert to list if numpy
            'action': int(action),
            'reward': float(reward)
        })

        state = next_state

    data_log.append(episode_data)

# Save all episodes to a JSON file
with open('snake_test_data.json', 'w') as f:
    json.dump(data_log, f, indent=2)

print("Saved test data to snake_test_data.json")
