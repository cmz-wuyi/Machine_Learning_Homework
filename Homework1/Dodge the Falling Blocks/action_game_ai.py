import pygame
import random
import numpy as np

# ==========================================
# 1. Game Configuration
# ==========================================
WIDTH = 400
HEIGHT = 400
PLAYER_SIZE = 50
ENEMY_SIZE = 50
FPS = 300  # Faster speed, convenient for training

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


# ==========================================
# 2. Q-Learning Agent (Supports Discretization)
# ==========================================
class QLearningAgent:
    def __init__(self):
        # Actions: 0=Stay, 1=Left, 2=Right
        self.actions = [0, 1, 2]
        self.q_table = {}
        self.epsilon = 1.0  # Initial: 100% random exploration
        self.epsilon_min = 0.01  # Minimum: keep 1% randomness
        self.epsilon_decay = 0.99  # Decay: decrease a little each episode
        self.alpha = 0.1
        self.gamma = 0.9

    def get_state_key(self, player_x, enemy_x, enemy_y):
        """
        Discretize continuous pixel coordinates into grid indices.
        Screen width is divided into 8 columns.
        """
        # Divide the 400-pixel-wide screen into 8 areas (each 50 pixels)
        p_idx = int(player_x // 50)
        e_x_idx = int(enemy_x // 50)
        e_y_idx = int(enemy_y // 50)

        # Status = (Where is the player, in which column is the enemy, and how tall is the enemy)
        return (p_idx, e_x_idx, e_y_idx)

    def get_action(self, state):
        # If this state hasn't been seen before, initialize to [0,0,0]
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Choose the action with the largest Q value
        return self.actions[np.argmax(self.q_table[state])]

    def update(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table: self.q_table[next_state] = [0.0, 0.0, 0.0]

        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        # Q-learning
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q


# ==========================================
# 3. Game Main Loop
# ==========================================
def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Option B: AI Dodge Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    agent = QLearningAgent()

    # Statistics
    episodes = 0
    scores = []

    # Game variables
    player_x = WIDTH // 2
    enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
    enemy_y = -ENEMY_SIZE
    score = 0

    running = True
    while running:
        # --- 1. Get current state ---
        state = agent.get_state_key(player_x, enemy_x, enemy_y)

        # --- 2. AI decides action ---
        # 0=Stay, 1=Left, 2=Right
        action = agent.get_action(state)

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 3. Execute action (Physics update) ---
        if action == 1:  # Left
            player_x -= 10
        elif action == 2:  # Right
            player_x += 10

        # Boundary limits
        if player_x < 0: player_x = 0
        if player_x > WIDTH - PLAYER_SIZE: player_x = WIDTH - PLAYER_SIZE

        # Enemy falls
        enemy_y += 15  # Fall speed

        # --- 4. Calculate reward and collision detection ---
        reward = 0
        done = False

        # Condition A: Dodge successful (enemy falls off screen)
        if enemy_y > HEIGHT:
            reward = 10
            score += 1
            # Reset enemy
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)

        # Condition B: Collision occurred
        player_rect = pygame.Rect(player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        enemy_rect = pygame.Rect(enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE)

        if player_rect.colliderect(enemy_rect):
            reward = -100
            done = True

        # Give a small reward for staying alive to encourage movement
        reward += 0.1

        # --- 5. AI Learning (Update) ---
        next_state = agent.get_state_key(player_x, enemy_x, enemy_y)
        agent.update(state, action, reward, next_state)

        # --- 6. Rendering ---
        screen.fill(WHITE)

        # Draw player (Blue)
        pygame.draw.rect(screen, BLUE, (player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE))
        # Draw enemy (Red)
        pygame.draw.rect(screen, RED, (enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE))

        # Display score
        text = font.render(f"Ep: {episodes} Score: {score}", True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)  # Control game speed

        # --- Episode end reset ---
        if done:
            episodes += 1
            scores.append(score)

            # --- New: Decrease Epsilon at the end of each episode ---
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # Print current score and Epsilon value for observation
            print(f"Episode {episodes}: Score = {score}, Epsilon = {agent.epsilon:.3f}")

            # Reset game state
            player_x = WIDTH // 2
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
            score = 0

            # Stop at 300 episodes
            if episodes >= 300:
                print("Training Finished!")
                running = False

    pygame.quit()


if __name__ == "__main__":
    run_game()