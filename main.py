import sys
import numpy as np
import random
import time
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QDoubleSpinBox, QLabel, QTextEdit, QSizePolicy, QSpinBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QPalette


def generate_random_maze(maze_size):
    """
    随机生成一个迷宫数组。
    :param maze_size: 迷宫的尺寸，生成迷宫尺寸为 maze_size * maze_size
    :return: 返回一个 maze_size * maze_size 的迷宫数组，1表示墙壁，0表示通道
    """
    # 初始化迷宫数组，1表示墙壁，0表示通道
    maze = np.ones((maze_size, maze_size), dtype=int)

    # 设置起点和终点
    start_point = (1, 1)
    destination = (maze_size - 2, maze_size - 2)

    # 将起点和终点设置为通道
    maze[start_point] = 0
    maze[destination] = 0

    # 使用深度优先搜索（DFS）算法生成迷宫
    stack = [start_point]
    visited = set()
    visited.add(start_point)

    while stack:
        current = stack[-1]
        x, y = current

        # 获取当前点的邻居
        neighbors = []
        if x > 1 and (x - 2, y) not in visited:
            neighbors.append((x - 2, y))
        if x < maze_size - 2 and (x + 2, y) not in visited:
            neighbors.append((x + 2, y))
        if y > 1 and (x, y - 2) not in visited:
            neighbors.append((x, y - 2))
        if y < maze_size - 2 and (x, y + 2) not in visited:
            neighbors.append((x, y + 2))

        if neighbors:
            next_cell = random.choice(neighbors)
            nx, ny = next_cell

            # 打通当前点和下一个点之间的墙
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0

            stack.append(next_cell)
            visited.add(next_cell)
        else:
            stack.pop()

    return maze





# AI智能体
class Agent:
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions

    def choose_action(self, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(Q_table[self.state[0], self.state[1], :])

    def update_state(self, new_state):
        self.state = new_state


# 更新AI智能体在迷宫中的位置
def update_agent(agent, action):
    row, col = agent.state
    if action == 0:  # 上
        row = max(row - 1, 0)
    elif action == 1:  # 下
        row = min(row + 1, maze_height - 1)
    elif action == 2:  # 左
        col = max(col - 1, 0)
    else:  # 右
        col = min(col + 1, maze_width - 1)
    new_state = (row, col)
    return new_state


# 不同的数字表示的方位
def getChinesefromNum(action):
    action_dict = {0: "上", 1: "下", 2: "左", 3: "右"}
    return action_dict.get(action, "")


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DQN迷宫寻路")
        self.setGeometry(100, 100, 400, 600)
        global maze
        # 主布局
        main_layout = QVBoxLayout()

        # 迷宫显示部分
        self.maze_widget = QWidget()
        self.maze_layout = QGridLayout()
        self.maze_buttons = [[QPushButton() for _ in range(maze_width)] for _ in range(maze_height)]

        # 初始化时一次性添加所有按钮
        for i in range(maze_height):
            for j in range(maze_width):
                btn = self.maze_buttons[i][j]
                self.maze_layout.addWidget(btn, i, j)

        self.maze_widget.setLayout(self.maze_layout)
        self.maze_layout.setSpacing(1)
        main_layout.addWidget(self.maze_widget)
        self.update_maze_ui()
        # 参数调整部分
        param_widget = QWidget()
        param_layout = QGridLayout()

        # num_episodes
        global num_episodes, epsilon, learning_rate, discount_factor
        self.episodes_label = QLabel("迭代次数:")
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 99999)
        self.episodes_spin.setValue(num_episodes)
        param_layout.addWidget(self.episodes_label, 0, 0)
        param_layout.addWidget(self.episodes_spin, 1, 0)

        # epsilon
        self.epsilon_label = QLabel("随机率:")
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setValue(epsilon)
        self.epsilon_spin.setRange(0, 1)
        self.epsilon_spin.setSingleStep(0.01)
        param_layout.addWidget(self.epsilon_label, 0, 1)
        param_layout.addWidget(self.epsilon_spin, 1, 1)

        # learning_rate
        self.lr_label = QLabel("学习率:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setValue(learning_rate)
        self.lr_spin.setRange(0, 1)
        self.lr_spin.setSingleStep(0.01)
        param_layout.addWidget(self.lr_label, 0, 2)
        param_layout.addWidget(self.lr_spin, 1, 2)

        # discount_factor
        self.df_label = QLabel("折扣因子:")
        self.df_spin = QDoubleSpinBox()
        self.df_spin.setValue(discount_factor)
        self.df_spin.setRange(0, 1)
        self.df_spin.setSingleStep(0.01)
        param_layout.addWidget(self.df_label, 0, 3)
        param_layout.addWidget(self.df_spin, 1, 3)

        # 按钮
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        param_layout.addWidget(self.train_button, 2, 0, 1, 2)

        self.animation_button = QPushButton("动画演示")
        self.animation_button.clicked.connect(self.start_animation)
        param_layout.addWidget(self.animation_button, 2, 2, 1, 2)

        param_widget.setLayout(param_layout)
        main_layout.addWidget(param_widget)

        # 训练日志部分
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        main_layout.addWidget(self.log_widget)

        # 设置主布局
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 训练相关变量
        self.agent = Agent((1, 1), [0, 1, 2, 3])
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_animation)

    def update_maze_ui(self):
        """
        更新迷宫UI，仅修改按钮的样式，不重新添加按钮。
        """
        for i in range(maze_height):
            for j in range(maze_width):
                btn = self.maze_buttons[i][j]
                if maze[i][j] == 1:
                    btn.setStyleSheet("background-color: black;")
                elif i == init_position["x"] and j == init_position["y"]:
                    btn.setStyleSheet("background-color: blue;")
                elif i == end_position["x"] and j == end_position["y"]:
                    btn.setStyleSheet("background-color: green;")
                else:
                    btn.setStyleSheet("background-color: white;")

    # Q-Learning算法
    def q_learning(self, agent, num_episodes, epsilon, learning_rate, discount_factor):
        global visualize
        for episode in range(num_episodes):
            agent.state = (init_position["x"], init_position["y"])  # 初始化智能体的状态为起始点
            score = 0
            steps = 0
            path = []
            while agent.state != (end_position["x"], end_position["y"]):  # 智能体到达目标点结束
                action = agent.choose_action(epsilon)
                new_state = update_agent(agent, action)

                path.append(getChinesefromNum(action))

                # 如果设置成-5，那么相比撞墙，他不会选择绕路绕5格以上的路（惩罚5以上）。
                # reward = -1 if maze[new_state] == 0 else -5  # 根据新状态更新奖励

                reward = -1 if maze[new_state] == 0 else -100  # 根据新状态更新奖励

                # 陷入局部最优
                # distance_to_goal = abs(new_state[0] - (maze_height - 1)) + abs(new_state[1] - (maze_width - 1))
                # reward = -distance_to_goal if maze[new_state] == 0 else -999999999999999  # 根据新状态更新奖励

                # reward = (0 - distance_to_goal / (maze_height + maze_width)) if maze[new_state] == 0 else -999  # 根据新状态更新奖励

                Q_table[agent.state[0], agent.state[1], action] += learning_rate * \
                                                                   (reward + discount_factor *
                                                                    np.max(Q_table[new_state]) - Q_table[
                                                                        agent.state[0], agent.state[1], action])
                agent.update_state(new_state)
                score += reward
                steps += 1

            # 输出当前的episode和最佳路径长度
            best_path_length = int(-score / 5)
            if episode % 10 == 0:
                # print(f"重复次数: {episode}, 路径长度: {steps}")
                # print(f"移动路径: {path}")
                self.log_widget.append(f"Episode: {episode} completed in {steps} steps, score:{-score}.")

    def start_training(self):
        global num_episodes, epsilon, learning_rate, discount_factor, maze_height, maze_width, maze

        num_episodes = self.episodes_spin.value()
        epsilon = self.epsilon_spin.value()
        learning_rate = self.lr_spin.value()
        discount_factor = self.df_spin.value()

        # 重新初始化 Q 表和 Agent
        global Q_table
        Q_table = np.zeros((maze_height, maze_width, 4))

        # 生成新的随机迷宫
        maze = generate_random_maze(maze_size)
        maze_height, maze_width = maze.shape
        # Q_table = np.zeros((maze_height, maze_width, 4))
        print(maze)
        # 更新迷宫UI
        self.update_maze_ui()
        QApplication.processEvents()
        self.agent.state = (init_position["x"], init_position["y"])  # 初始化智能体的状态为起始点
        # 运行Q-Learning算法
        self.q_learning(self.agent, num_episodes, epsilon, learning_rate, discount_factor)

    def start_animation(self):
        global maze, maze_height, maze_width, init_position, end_position, Q_table


        # 重置 agent 到起点
        self.agent.state = (init_position["x"], init_position["y"])  # 初始化智能体的状态为起始点
        self.timer.start(100)  # 每 100 毫秒一步

    def step_animation(self):
        if self.agent.state != (end_position["x"], end_position["y"]):  # 智能体到达目标点结束
            # 重置迷宫颜色
            for i in range(maze_height):
                for j in range(maze_width):
                    btn = self.maze_buttons[i][j]
                    if maze[i][j] == 1:
                        btn.setStyleSheet("background-color: black;")
                    elif i == init_position["x"] and j == init_position["y"]:
                        btn.setStyleSheet("background-color: blue;")
                    elif i == end_position["x"] and j == end_position["y"]:
                        btn.setStyleSheet("background-color: green;")
                    else:
                        btn.setStyleSheet("background-color: white;")
            action = np.argmax(
                Q_table[self.agent.state[0], self.agent.state[1], :])  # 根据Q值表选择最优动作
            new_state = update_agent(self.agent, action)
            # 更新新的位置颜色
            ni, nj = new_state
            self.maze_buttons[ni][nj].setStyleSheet("background-color: red;")

            self.agent.update_state(new_state)


# 主程序
if __name__ == "__main__":
    num_episodes = 600  # 迭代次数
    epsilon = 0.1  # 随机率
    learning_rate = 0.1  # 学习率
    discount_factor = 0.9
    # 迷宫尺寸
    maze_size = 10
    maze = generate_random_maze(maze_size)
    maze_height, maze_width = maze.shape
    # 起始位置
    init_position = {"x": 1, "y": 1}
    end_position = {"x": maze_height - 2, "y": maze_width - 2}
    # 初始化Q值表
    Q_table = np.zeros((maze_height, maze_width, 4))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())