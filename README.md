# SnakeGame
The files classifier.py, layers.py, optim.py are used to code model.py without needing to use pytorch. 
Inside of model.py there are two classes: Linear_QNet (neural network) and QTrainer (trainer class that performs the Q-Learning Update rule)

Linear_QNet 
- Implements a fully connected neural network with one hidden layer. The parameters are initialized internally already, so the constructor can be called without arguments. 
- Handles forward pass computations (used to predict Q-values) and backward pass (used to compute gradients for weights, bias, activations).
- Provides a save() and load() file so the model can be stored in a .pkl file and restored later.

QTrainer 
- Converts the current state and next state into NumPy arrays and feeds them through the model to predict the Q-values. 
- Uses the Bellman Equation to calculate the target Q-value based on the action taken and the current state: 
    Qnew = reward[idx] + self.gamma * np.max(next_q)
    - reward[idx] is the reward received for this experience.
    - gamma determines how much the agent values future rewards compared to immediate rewards.
    - np.max(next_q) looks at all possible actions in the next state and chooses the one with the highest predicted reward.
- Compute the MSE by measuring the difference between the Target Q-value and Predicted Q-value. 
- Performs a backward pass and applies gradient descent to update model's weights. 

## snake.py
snake.py is used to run the actual snake game using pygame. The game is run locally by running snake.py and using the arrow keys to navigate the snake within the window and collect food. If the snake hits the edge of the window or runs into itself, the game ends and allows the user to restart using the arrow keys or quit entirely by pressing backspace.

