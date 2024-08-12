import tkinter as tk
from tkinter import messagebox
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Constants
CELL_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (173, 216, 230)

# Global variables for user placements
goal_pos = None
coins_pos = []
slippery_pos = None
walls_pos = []
clock_pos = None
placing = "goal"
num_coins = 0
delay = 100
ALL_POSSIBLE_ACTIONS = ["up", "down", "left", "right"]

# Function to start the game with user-defined parameters
def start_game():
    try:
        global GRID_SIZE, pre_train_episodes, max_steps_per_episode, alpha, gamma, epsilon
        global num_coins, coin_reward, algorithm, train_episodes, delay

        GRID_SIZE = int(grid_size_entry.get())
        pre_train_episodes = int(pre_train_episodes_entry.get())
        max_steps_per_episode = int(max_steps_per_episode_entry.get())
        alpha = float(alpha_entry.get())
        gamma = float(gamma_entry.get())
        epsilon = float(epsilon_entry.get())
        num_coins = int(num_coins_entry.get())
        coin_reward = int(coin_reward_entry.get())
        algorithm = algorithm_var.get()
        train_episodes = int(train_episodes_entry.get())
        delay = int(delay_entry.get())

        window.destroy()
        place_elements()
    except ValueError:
        tk.messagebox.showerror("Input Error", "Please enter valid values for all fields.")

# Pygame element placement
def place_elements():
    global placing, goal_pos, coins_pos, slippery_pos, walls_pos, clock_pos

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 50))
    pygame.display.set_caption('Place Elements')

    # Load images
    agent_img = pygame.image.load("agent.png")
    coin_img = pygame.image.load("coin.png")
    goal_img = pygame.image.load("goal.png")
    slippery_img = pygame.image.load("slippery.png")
    police_img = pygame.image.load("police.png")
    wall_img = pygame.image.load("wall.png")
    clock_img = pygame.image.load("clock.png")

    agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
    coin_img = pygame.transform.scale(coin_img, (CELL_SIZE, CELL_SIZE))
    goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
    slippery_img = pygame.transform.scale(slippery_img, (CELL_SIZE, CELL_SIZE))
    police_img = pygame.transform.scale(police_img, (CELL_SIZE, CELL_SIZE))
    wall_img = pygame.transform.scale(wall_img, (CELL_SIZE, CELL_SIZE))
    clock_img = pygame.transform.scale(clock_img, (CELL_SIZE, CELL_SIZE))

    font = pygame.font.SysFont(None, 24)

    def draw_grid():
        screen.fill(WHITE)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, BLACK, rect, 1)
                if [x, y] == goal_pos:
                    screen.blit(goal_img, rect)
                elif [x, y] in coins_pos:
                    screen.blit(coin_img, rect)
                elif [x, y] == slippery_pos:
                    screen.blit(slippery_img, rect)
                elif [x, y] in walls_pos:
                    screen.blit(wall_img, rect)
                elif [x, y] == clock_pos:
                    screen.blit(clock_img, rect)
                elif [x, y] == [0, 0]:
                    screen.blit(agent_img, rect)

        # Draw the text label at the bottom
        if placing == "goal":
            label = font.render("Place the Goal", True, BLACK)
        elif placing == "coins":
            label = font.render(f"Place Coins ({len(coins_pos)}/{num_coins})", True, BLACK)
        elif placing == "slippery":
            label = font.render("Place the Slippery Cell", True, BLACK)
        elif placing == "walls":
            label = font.render("Place the Walls", True, BLACK)
        elif placing == "clock":
            label = font.render("Place the Clock", True, BLACK)
        screen.blit(label, (10, GRID_SIZE * CELL_SIZE + 10))

    running = True
    while running:
        draw_grid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                grid_pos = [x // CELL_SIZE, y // CELL_SIZE]
                if event.button == 1:  # Left click to place element
                    if placing == "goal":
                        goal_pos = grid_pos
                    elif placing == "coins" and len(coins_pos) < num_coins:
                        if grid_pos not in coins_pos:
                            coins_pos.append(grid_pos)
                    elif placing == "slippery":
                        slippery_pos = grid_pos
                    elif placing == "walls":
                        if grid_pos not in walls_pos:
                            walls_pos.append(grid_pos)
                    elif placing == "clock":
                        clock_pos = grid_pos
                elif event.button == 3:  # Right click to move to the next element
                    if placing == "goal":
                        placing = "coins"
                    elif placing == "coins" and len(coins_pos) >= num_coins:
                        placing = "slippery"
                    elif placing == "slippery":
                        placing = "walls"
                    elif placing == "walls":
                        placing = "clock"
                    elif placing == "clock":
                        running = False
        pygame.display.flip()

    pygame.quit()
    pygame_game()

# Define the pygame_game function here
def pygame_game():
    # Initialize Pygame
    pygame.init()

    SCREEN_SIZE = CELL_SIZE * GRID_SIZE

    # Load images
    agent_img = pygame.image.load("agent.png")
    coin_img = pygame.image.load("coin.png")
    goal_img = pygame.image.load("goal.png")
    slippery_img = pygame.image.load("slippery.png")
    police_img = pygame.image.load("police.png")
    wall_img = pygame.image.load("wall.png")
    clock_img = pygame.image.load("clock.png")

    agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
    coin_img = pygame.transform.scale(coin_img, (CELL_SIZE, CELL_SIZE))
    goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
    slippery_img = pygame.transform.scale(slippery_img, (CELL_SIZE, CELL_SIZE))
    police_img = pygame.transform.scale(police_img, (CELL_SIZE, CELL_SIZE))
    wall_img = pygame.transform.scale(wall_img, (CELL_SIZE, CELL_SIZE))
    clock_img = pygame.transform.scale(clock_img, (CELL_SIZE, CELL_SIZE))

    # Initialize the environment
    class GridWorld:
        def __init__(self, grid_size, num_coins, goal_pos, coins_pos, slippery_pos, walls_pos, clock_pos):
            self.grid_size = grid_size
            self.num_coins = num_coins
            self.goal_pos = goal_pos
            self.coins_pos = coins_pos
            self.slippery_pos = slippery_pos
            self.walls_pos = walls_pos
            self.clock_pos = clock_pos
            self.reset()

        def reset(self):
            self.agent_pos = [0, 0]
            self.coins = self.coins_pos[:]
            self.slippery = self.slippery_pos
            self.walls = self.walls_pos[:]
            self.clock = self.clock_pos
            self.police_pos, self.police_direction = self.random_police_pos_and_direction()
            self.clock_active = False
            self.stop_police = False
            self.visited = set()
            return self.get_state() # Updated to return detailed state

        # Include detailed state information
        def get_state(self):
            state = np.zeros((self.grid_size, self.grid_size, 4))  # Updated for goal, coins, and clock
            state[self.agent_pos[0], self.agent_pos[1], 0] = 1  # Agent's position
            if self.goal_pos:
                state[self.goal_pos[0], self.goal_pos[1], 1] = 1  # Goal's position
            for coin in self.coins:
                state[coin[0], coin[1], 2] = 1  # Coins' positions
            if self.clock:
                state[self.clock[0], self.clock[1], 3] = 1  # Clock's position
            return state

        def step(self, action):
            # Determine the next position based on the action
            if action == "up" and self.agent_pos[1] > 0:
                next_pos = [self.agent_pos[0], self.agent_pos[1] - 1]
            elif action == "down" and self.agent_pos[1] < self.grid_size - 1:
                next_pos = [self.agent_pos[0], self.agent_pos[1] + 1]
            elif action == "left" and self.agent_pos[0] > 0:
                next_pos = [self.agent_pos[0] - 1, self.agent_pos[1]]
            elif action == "right" and self.agent_pos[0] < self.grid_size - 1:
                next_pos = [self.agent_pos[0] + 1, self.agent_pos[1]]
            else:
                next_pos = self.agent_pos  # No movement if action is invalid

            # Check if the next position is a wall
            if next_pos not in self.walls:
                self.agent_pos = next_pos  # Update position only if it's not a wall
            # The cost of every move (Energy/Fuel)
            reward = -1
            done = False

            state_action_pair = (tuple(self.agent_pos), action)
            if state_action_pair in self.visited:
                reward -= 20  # Penalty for redundant movements
            else:
                self.visited.add(state_action_pair)

            if self.agent_pos == self.goal_pos:
                reward = 100
                done = True

            if self.agent_pos in self.coins:
                reward += coin_reward
                self.coins.remove(self.agent_pos)

            if self.agent_pos == self.slippery:
                if random.random() < 0.5:
                    self.agent_pos = random.choice(self.valid_moves(self.agent_pos))

            if self.agent_pos == self.clock:
                self.clock_active = True
                self.stop_police = True
                self.clock = None  # Remove the clock from the grid
                reward += 5

            if self.agent_pos == self.police_pos and not self.clock_active:
                reward -= 50
                done = True

            return self.get_state(), reward, done

        def valid_moves(self, pos):
            moves = []
            if pos[1] > 0:
                moves.append([pos[0], pos[1] - 1])
            if pos[1] < self.grid_size - 1:
                moves.append([pos[0], pos[1] + 1])
            if pos[0] > 0:
                moves.append([pos[0] - 1, pos[1]])
            if pos[0] < self.grid_size - 1:
                moves.append([pos[0] + 1, pos[1]])
            return moves

        def random_police_pos_and_direction(self):
            # Ensure the police can move across the entire row or column
            if random.choice(["horizontal", "vertical"]) == "horizontal":
                pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
                direction = random.choice(["left", "right"])
            else:
                pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
                direction = random.choice(["up", "down"])
            return pos, direction

        def move_police(self):
            if self.stop_police:
                return

            next_pos = self.police_pos[:]
            deflect = False

            if self.police_direction in ["left", "right"]:
                if self.police_direction == "left":
                    next_pos[0] -= 1
                    if next_pos[0] < 0 or next_pos in self.walls or next_pos == self.slippery_pos:
                        self.police_direction = "right"
                        next_pos[0] = self.police_pos[0] + 1
                elif self.police_direction == "right":
                    next_pos[0] += 1
                    if next_pos[0] >= self.grid_size or next_pos in self.walls or next_pos == self.slippery_pos:
                        self.police_direction = "left"
                        next_pos[0] = self.police_pos[0] - 1
            elif self.police_direction in ["up", "down"]:
                if self.police_direction == "up":
                    next_pos[1] -= 1
                    if next_pos[1] < 0 or next_pos in self.walls or next_pos == self.slippery_pos:
                        self.police_direction = "down"
                        next_pos[1] = self.police_pos[1] + 1
                elif self.police_direction == "down":
                    next_pos[1] += 1
                    if next_pos[1] >= self.grid_size or next_pos in self.walls or next_pos == self.slippery_pos:
                        self.police_direction = "up"
                        next_pos[1] = self.police_pos[1] - 1

            if not deflect:
                self.police_pos = next_pos

    # Q-learning algorithm
    class QLearningAgent:
        def __init__(self, grid_size, alpha, gamma, epsilon, epsilon_decay):
            self.q_table = np.zeros((grid_size, grid_size, 4))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay

        def choose_action(self, state):
            agent_pos = self.get_agent_position(state)
            if random.random() < self.epsilon:
                return random.choice(["up", "down", "left", "right"])  # Explore: choose a random action
            else:
                return ["up", "down", "left", "right"][
                    np.argmax(self.q_table[agent_pos[0], agent_pos[1]])]  # Exploit: choose the best known action

        def update(self, state, action, reward, next_state):
            agent_pos = self.get_agent_position(state)
            next_agent_pos = self.get_agent_position(next_state)
            action_idx = ["up", "down", "left", "right"].index(action)
            next_max = np.max(self.q_table[next_agent_pos[0], next_agent_pos[1]])
            current_q = self.q_table[agent_pos[0], agent_pos[1], action_idx]
            self.q_table[agent_pos[0], agent_pos[1], action_idx] = current_q + self.alpha * (
                    reward + self.gamma * next_max - current_q)
            print(f"Updated Q-table at state: {agent_pos}, action: {action} with reward: {reward}")

        def get_agent_position(self, state):
            return np.argwhere(state[:, :, 0] == 1)[0]  # Assuming the first channel represents the agent's position

        def decay_epsilon(self):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    # SARSA algorithm
    class SARSAAgent:
        def __init__(self, grid_size, alpha, gamma, epsilon, epsilon_decay):
            self.q_table = np.zeros((grid_size, grid_size, 4))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay

        def choose_action(self, state):
            agent_pos = self.get_agent_position(state)
            if random.random() < self.epsilon:
                return random.choice(["up", "down", "left", "right"])
            else:
                return ["up", "down", "left", "right"][np.argmax(self.q_table[agent_pos[0], agent_pos[1]])]

        def update(self, state, action, reward, next_state, next_action):
            agent_pos = self.get_agent_position(state)
            next_agent_pos = self.get_agent_position(next_state)
            action_idx = ["up", "down", "left", "right"].index(action)
            next_action_idx = ["up", "down", "left", "right"].index(next_action)
            current_q = self.q_table[agent_pos[0], agent_pos[1], action_idx]
            next_q = self.q_table[next_agent_pos[0], next_agent_pos[1], next_action_idx]
            self.q_table[agent_pos[0], agent_pos[1], action_idx] = current_q + self.alpha * (
                    reward + self.gamma * next_q - current_q)
            print(f"Updated Q-table at state: {agent_pos}, action: {action} with reward: {reward}")

        def get_agent_position(self, state):
            return np.argwhere(state[:, :, 0] == 1)[0]  # Assuming the first channel represents the agent's position

        def decay_epsilon(self):
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    # Initialize the environment and agent
    env = GridWorld(GRID_SIZE, num_coins, goal_pos, coins_pos, slippery_pos, walls_pos, clock_pos)
    epsilon_decay = 0.995
    if algorithm == "q_learning":
        agent = QLearningAgent(GRID_SIZE, alpha, gamma, epsilon, epsilon_decay)
    else:
        agent = SARSAAgent(GRID_SIZE, alpha, gamma, epsilon, epsilon_decay)

    # Pre-Training phase
    pre_train_rewards = []
    for episode in range(pre_train_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            if algorithm == "q_learning":
                agent.update(state, action, reward, next_state)
            else:
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
            state = next_state
            total_reward += reward
            if done:
                break
        pre_train_rewards.append(total_reward)
        agent.decay_epsilon()
        print(f"Pre-Train Episode {episode + 1} finished with total reward: {total_reward}")

    # Scatter plot of learning progress
    plt.scatter(range(pre_train_episodes), pre_train_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Pre-Train Learning Progress')
    plt.show()

    def reward_plot_window():
        reward_window = tk.Tk()
        reward_window.title("Reward Plot")

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [], c='b')
        ax.set_xlim(0, train_episodes)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards')

        canvas = FigureCanvasTkAgg(fig, master=reward_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        return reward_window, scatter, ax

    reward_window, scatter, ax = reward_plot_window()

    def update_plot(rewards):
        scatter.set_offsets(np.c_[range(len(rewards)), rewards])
        ax.set_xlim(0, len(rewards))
        ax.set_ylim(min(rewards) - 10, max(rewards) + 10)
        plt.draw()
        reward_window.update()

    # Pygame animation for the gameplay phase
    def draw_grid(screen, env):
        screen.fill(WHITE)
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, BLACK, rect, 1)
                if [x, y] == env.agent_pos:
                    screen.blit(agent_img, rect)
                elif [x, y] == env.goal_pos:
                    screen.blit(goal_img, rect)
                elif [x, y] == env.slippery:
                    screen.blit(slippery_img, rect)
                elif [x, y] == env.clock:
                    screen.blit(clock_img, rect)
                elif [x, y] in env.coins:
                    screen.blit(coin_img, rect)
                elif [x, y] in env.walls:
                    screen.blit(wall_img, rect)
                elif [x, y] == env.police_pos:
                    screen.blit(police_img, rect)

    def run_animation():
        global delay
        screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption('Grid World RL')

        rewards = []
        for episode in range(train_episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                draw_grid(screen, env)
                pygame.display.flip()
                pygame.time.wait(delay)
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                env.move_police()
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
            update_plot(rewards)
            print(f"Gameplay Episode {episode + 1} finished with total reward: {total_reward}")

        pygame.quit()

    animation_thread = threading.Thread(target=run_animation)
    animation_thread.start()
    reward_window.mainloop()

# Initialize tkinter window
window = tk.Tk()
window.title("Grid World RL Settings")

# Define input fields
tk.Label(window, text="Grid Size:").grid(row=0, column=0)
grid_size_entry = tk.Entry(window)
grid_size_entry.grid(row=0, column=1)
grid_size_entry.insert(tk.END, "7")

tk.Label(window, text="Pre train episodes:").grid(row=1, column=0)
pre_train_episodes_entry = tk.Entry(window)
pre_train_episodes_entry.grid(row=1, column=1)
pre_train_episodes_entry.insert(tk.END, "1000")

tk.Label(window, text="Max steps per episode:").grid(row=2, column=0)
max_steps_per_episode_entry = tk.Entry(window)
max_steps_per_episode_entry.grid(row=2, column=1)
max_steps_per_episode_entry.insert(tk.END, "75")

tk.Label(window, text="Alpha:").grid(row=3, column=0)
alpha_entry = tk.Entry(window)
alpha_entry.grid(row=3, column=1)
alpha_entry.insert(tk.END, "0.1")

tk.Label(window, text="Gamma:").grid(row=4, column=0)
gamma_entry = tk.Entry(window)
gamma_entry.grid(row=4, column=1)
gamma_entry.insert(tk.END, "0.9")

tk.Label(window, text="Epsilon:").grid(row=5, column=0)
epsilon_entry = tk.Entry(window)
epsilon_entry.grid(row=5, column=1)
epsilon_entry.insert(tk.END, "0.1")

tk.Label(window, text="Num Coins:").grid(row=7, column=0)
num_coins_entry = tk.Entry(window)
num_coins_entry.grid(row=7, column=1)
num_coins_entry.insert(tk.END, "3")

tk.Label(window, text="Coin_reward:").grid(row=8, column=0)
coin_reward_entry = tk.Entry(window)
coin_reward_entry.grid(row=8, column=1)
coin_reward_entry.insert(tk.END, "10")

tk.Label(window, text="Train_episodes:").grid(row=9, column=0)
train_episodes_entry = tk.Entry(window)
train_episodes_entry.grid(row=9, column=1)
train_episodes_entry.insert(tk.END, "100")

tk.Label(window, text="Algorithm:").grid(row=10, column=0)
algorithm_var = tk.StringVar(window)
algorithm_var.set("q_learning")
tk.OptionMenu(window, algorithm_var, "q_learning", "sarsa").grid(row=10, column=1)

tk.Label(window, text="Delay:").grid(row=11, column=0)
delay_entry = tk.Entry(window)
delay_entry.grid(row=11, column=1)
delay_entry.insert(tk.END, "100")

tk.Button(window, text="Start Game", command=start_game).grid(row=12, column=0, columnspan=2)

# Run the tkinter window to get user inputs
window.mainloop()