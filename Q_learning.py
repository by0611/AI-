import numpy as np
import random

# 定義迷宮，'S' = 起點，'G' = 終點，0 = 通道，1 = 牆壁
maze = np.array([
    ['S', 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1 , 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0 , 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0 , 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [0 , 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1 , 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0 , 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0 , 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0 , 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1 , 1, 0, 0, 1, 1, 1, 1, 1, 'G']
])

# 位置轉換函數
def pos_to_state(r, c): return r * 10 + c
def state_to_pos(s): return (s // 10, s % 10)

# 找出起點與終點，並轉成整數
start, goal = None, None
for r in range(10):
    for c in range(10):
        if maze[r, c] == 'S':
            start = pos_to_state(r, c)
            maze[r, c] = 0
        elif maze[r, c] == 'G':
            goal = pos_to_state(r, c)
            maze[r, c] = 0
maze = maze.astype(int)

# 初始化 Reward 矩陣
R = np.full((100, 100), -np.inf)
directions = [(-1,0), (1,0), (0,-1), (0,1)]

for r in range(10):
    for c in range(10):
        if maze[r, c] == 0:
            from_state = pos_to_state(r, c)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10 and maze[nr, nc] == 0:
                    to_state = pos_to_state(nr, nc)
                    R[from_state, to_state] = 100 if to_state == goal else -1

# 初始化 Q-table
Q = np.zeros((100, 100))

# Q-learning 參數
gamma = 0.8
alpha = 0.9
epsilon = 0.2
episodes = 1000

# 訓練
for _ in range(episodes):
    state = random.choice([s for s in range(100) if not np.all(np.isinf(R[s]))])
    while state != goal:
        actions = [a for a in range(100) if not np.isinf(R[state, a])]
        if not actions:
            break
        if random.random() < epsilon:
            next_state = random.choice(actions)
        else:
            next_state = max(actions, key=lambda a: Q[state, a])
        future_actions = [a for a in range(100) if not np.isinf(R[next_state, a])]
        max_q = max([Q[next_state, a] for a in future_actions], default=0)
        Q[state, next_state] += alpha * (R[state, next_state] + gamma * max_q - Q[state, next_state])
        state = next_state

# 找到最短路徑
def extract_path(start, goal):
    path = [start]
    current = start
    visited = set()
    
    while current != goal:
        visited.add(current)
        next_states = [s for s in range(100) if not np.isinf(R[current, s])]
        if not next_states:
            break
        next_state = max(next_states, key=lambda s: Q[current, s])
        
        if next_state in visited or np.isinf(R[current, next_state]):
            break
        
        path.append(next_state)
        current = next_state
        
        if len(path) > 100:
            break
    return path

path = extract_path(start, goal)

# 輸出結果
print("\n最短路徑 (state index):", path)
print("座標路徑:", [state_to_pos(s) for s in path])

# 顯示迷宮路徑（可視化）
display_maze = maze.astype(str)
display_maze[display_maze == '1'] = '█'
display_maze[display_maze == '0'] = ' '

# 標記路徑
for s in path:
    r, c = state_to_pos(s)
    if (r, c) != state_to_pos(start) and (r, c) != state_to_pos(goal):  # 確保起點與終點不被覆蓋
        display_maze[r, c] = '*'
display_maze[state_to_pos(start)] = 'S'
display_maze[state_to_pos(goal)] = 'G'

print("\n迷宮最短路徑(文字版):")
for row in display_maze:
    print(' '.join(row))
