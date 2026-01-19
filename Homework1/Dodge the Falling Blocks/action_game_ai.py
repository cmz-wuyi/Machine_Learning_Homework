import pygame
import random
import numpy as np

WIDTH = 400
HEIGHT = 400
PLAYER_SIZE = 50
ENEMY_SIZE = 50
FPS = 300

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class QLearningAgent:
    def __init__(self):
        self.actions = [0, 1, 2]
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.alpha = 0.1
        self.gamma = 0.9

    def get_state_key(self, player_x, enemy_x, enemy_y):
        p_idx = int(player_x // 50)
        e_x_idx = int(enemy_x // 50)
        e_y_idx = int(enemy_y // 50)
        return (p_idx, e_x_idx, e_y_idx)

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return self.actions[np.argmax(self.q_table[state])]

    def update(self, state, action, reward, next_state):
        if state not in self.q_table: self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table: self.q_table[next_state] = [0.0, 0.0, 0.0]

        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q

def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Option B: AI Dodge Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    agent = QLearningAgent()

    episodes = 0
    scores = []

    player_x = WIDTH // 2
    enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
    enemy_y = -ENEMY_SIZE
    score = 0

    running = True
    while running:
        state = agent.get_state_key(player_x, enemy_x, enemy_y)

        action = agent.get_action(state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if action == 1:
            player_x -= 10
        elif action == 2:
            player_x += 10

        if player_x < 0: player_x = 0
        if player_x > WIDTH - PLAYER_SIZE: player_x = WIDTH - PLAYER_SIZE

        enemy_y += 15

        reward = 0
        done = False

        if enemy_y > HEIGHT:
            reward = 10
            score += 1
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)

        player_rect = pygame.Rect(player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        enemy_rect = pygame.Rect(enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE)

        if player_rect.colliderect(enemy_rect):
            reward = -100
            done = True

        reward += 0.1

        next_state = agent.get_state_key(player_x, enemy_x, enemy_y)
        agent.update(state, action, reward, next_state)

        screen.fill(WHITE)

        pygame.draw.rect(screen, BLUE, (player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE))
        pygame.draw.rect(screen, RED, (enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE))

        text = font.render(f"Ep: {episodes} Score: {score}", True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

        if done:
            episodes += 1
            scores.append(score)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            print(f"Episode {episodes}: Score = {score}, Epsilon = {agent.epsilon:.3f}")

            player_x = WIDTH // 2
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
            score = 0

            if episodes >= 300:
                print("Training Finished!")
                running = False

    pygame.quit()

if __name__ == "__main__":
    run_game()