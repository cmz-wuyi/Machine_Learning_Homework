import pygame
import random
import numpy as np

# ==========================================
# 1. 游戏配置
# ==========================================
WIDTH = 400
HEIGHT = 400
PLAYER_SIZE = 50
ENEMY_SIZE = 50
FPS = 300  # 速度快一点，方便训练

# 颜色
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


# ==========================================
# 2. Q-Learning 智能体 (支持离散化)
# ==========================================
class QLearningAgent:
    def __init__(self):
        # 动作: 0=不动, 1=左, 2=右
        self.actions = [0, 1, 2]
        self.q_table = {}
        self.epsilon = 1.0  # 初始：100% 随机探索 (像个婴儿)
        self.epsilon_min = 0.01  # 最低：保留 1% 的随机性
        self.epsilon_decay = 0.99  # 衰减：每回合减少一点点
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
        # 如果这个状态以前没见过，初始化为 [0,0,0]
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # 选 Q 值最大的动作
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
# 3. 游戏主循环
# ==========================================
def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Option B: AI Dodge Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    agent = QLearningAgent()

    # 统计数据
    episodes = 0
    scores = []

    # 游戏变量
    player_x = WIDTH // 2
    enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
    enemy_y = -ENEMY_SIZE
    score = 0

    running = True
    while running:
        # --- 1. 获取当前状态 ---
        state = agent.get_state_key(player_x, enemy_x, enemy_y)

        # --- 2. AI 决定动作 ---
        # 0=不动, 1=左, 2=右
        action = agent.get_action(state)

        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 3. 执行动作 (物理更新) ---
        if action == 1:  # 左
            player_x -= 10
        elif action == 2:  # 右
            player_x += 10

        # 边界限制
        if player_x < 0: player_x = 0
        if player_x > WIDTH - PLAYER_SIZE: player_x = WIDTH - PLAYER_SIZE

        # 敌人下落
        enemy_y += 15  # 下落速度

        # --- 4. 计算奖励与碰撞检测 ---
        reward = 0
        done = False

        # 判定 A: 躲避成功 (敌人掉出屏幕)
        if enemy_y > HEIGHT:
            reward = 10
            score += 1
            # 重置敌人
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)

        # 判定 B: 发生碰撞
        player_rect = pygame.Rect(player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        enemy_rect = pygame.Rect(enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE)

        if player_rect.colliderect(enemy_rect):
            reward = -100
            done = True

        # 稍微给一点活着的奖励，鼓励它动起来
        reward += 0.1

        # --- 5. AI 学习 (Update) ---
        next_state = agent.get_state_key(player_x, enemy_x, enemy_y)
        agent.update(state, action, reward, next_state)

        # --- 6. 绘图 (Rendering) ---
        screen.fill(WHITE)

        # 画玩家 (蓝色)
        pygame.draw.rect(screen, BLUE, (player_x, HEIGHT - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE))
        # 画敌人 (红色)
        pygame.draw.rect(screen, RED, (enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE))

        # 显示分数
        text = font.render(f"Ep: {episodes} Score: {score}", True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)  # 控制游戏速度

        # --- 回合结束重置 ---
        if done:
            episodes += 1
            scores.append(score)

            # --- 新增：每回合结束，让 AI 变得更自信一点 ---
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            # 打印当前分数和 Epsilon 值，方便观察
            print(f"Episode {episodes}: Score = {score}, Epsilon = {agent.epsilon:.3f}")

            # 重置游戏状态
            player_x = WIDTH // 2
            enemy_y = -ENEMY_SIZE
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
            score = 0

            # 到 300 回合
            if episodes >= 300:
                print("Training Finished!")
                running = False

    pygame.quit()


if __name__ == "__main__":
    run_game()