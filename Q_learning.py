import numpy as np
import matplotlib.pyplot as plt

# 定義迷宮
# 0: 可通行路徑
# 1: 牆壁
# 2: 目標位置
maze = np.array([
    [0,0,0,1,0,0,0,0,0,0],
    [1,1,0,1,0,1,1,1,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,1,1,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,1],
    [0,0,0,1,0,0,0,0,0,0],
    [1,1,0,0,1,1,1,1,1,2]   # 目標位置
])

# 取得迷宮的尺寸
N = maze.shape[0]

# 定義目標點
goal_pos = tuple(np.argwhere(maze == 2)[0])

# 定義可能的移動方向 (上, 下, 左, 右)
dirs = [(-1,0), (1,0), (0,-1), (0,1)]

# 建立獎勵 (R) 矩陣
# -1 代表牆壁 (不可通行)
# 100 代表目標點
# 0 代表普通可通行路徑
R = np.zeros((N,N), dtype=int)
for r in range(N):
    for c in range(N):
        if maze[r,c] == 1: # 牆壁
            R[r,c] = -1
        else:
            reward = 0
            # 檢查相鄰格子是否為目標點，如果是則給予高獎勵
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    if (nr, nc) == goal_pos:
                        reward = 100
                        break
            R[r,c] = reward
R[goal_pos] = 100 # 確保目標點本身的獎勵為100

# 初始化 Q 矩陣 (全部為 0)
Q = np.zeros_like(R, dtype=float)

# 設定 Q-learning 參數
gamma = 0.8 # 折扣因子 (discount factor) - 未來獎勵的重要性
alpha = 0.9 # 學習率 (learning rate) - 學習新資訊的速度
num_episodes = 2000 # 最大訓練回合數 (即使觸發提前終止也會停止)
patience = 10 # 短路徑連續沒有改善的容忍次數

# 函數：根據目前的 Q 矩陣尋找從起點到目標點的路徑
def find_path(q_matrix, start, goal, maze_rewards, max_steps=1000):
    current = start
    path = [current]
    steps = 0
    
    while current != goal and steps < max_steps:
        i, j = current
        next_moves = []
        
        # 檢查所有可能的移動方向
        for dr, dc in dirs:
            ni, nj = i + dr, j + dc
            # 確保新位置在迷宮範圍內且不是牆壁
            if 0 <= ni < N and 0 <= nj < N and maze_rewards[ni, nj] != -1:
                next_moves.append(((ni, nj), q_matrix[ni, nj]))
        
        if not next_moves: # 如果沒有可移動的路徑，表示受困
            return [], -1
        
        # 從可移動的路徑中選擇 Q 值最高的行動
        # 如果有多個 Q 值相同的最佳行動，隨機選擇一個
        max_q = -np.inf
        best_moves = []
        for move, q_val in next_moves:
            if q_val > max_q:
                max_q = q_val
                best_moves = [move]
            elif q_val == max_q:
                best_moves.append(move)

        if not best_moves: # 應該不會發生如果 next_moves 不為空
            return [], -1
        
        # 更新目前位置
        current = best_moves[np.random.choice(len(best_moves))] 
        path.append(current)
        steps += 1
    
    if current == goal: # 如果找到目標
        return path, steps
    else: # 如果超出最大步數或受困
        return [], -1

print("開始 Q-learning 訓練...")

# 使用 (0,0) 作為預設訓練起點，以評估模型收斂
training_start_pos = (0, 0) 

shortest_path_found = []
min_steps = float('inf') # 記錄找到的最短路徑步數
steps_per_episode = []   # 記錄每個回合找到路徑的步數
no_improvement_count = 0 # 連續沒有改善的計數器

# Q-learning 訓練迴圈
for episode in range(num_episodes):
    # 這個內部迴圈會遍歷迷宮中的所有可通行狀態，並更新其 Q 值
    for r_idx in range(N):
        for c_idx in range(N):
            if R[r_idx, c_idx] == -1: # 跳過牆壁
                continue
            
            reward = R[r_idx, c_idx]
            
            neighbors_q = []
            # 獲取所有相鄰可通行狀態的 Q 值
            if r_idx > 0 and R[r_idx-1, c_idx] != -1: # 上
                neighbors_q.append(Q[r_idx-1, c_idx])
            if r_idx < N-1 and R[r_idx+1, c_idx] != -1: # 下
                neighbors_q.append(Q[r_idx+1, c_idx])
            if c_idx > 0 and R[r_idx, c_idx-1] != -1: # 左
                neighbors_q.append(Q[r_idx, c_idx-1])
            if c_idx < N-1 and R[r_idx, c_idx+1] != -1: # 右
                neighbors_q.append(Q[r_idx, c_idx+1])
            
            # 取得鄰居中最大的 Q 值
            max_q_next = max(neighbors_q) if neighbors_q else 0
            
            # Q-learning 更新公式
            Q[r_idx, c_idx] = (1 - alpha) * Q[r_idx, c_idx] + alpha * (reward + gamma * max_q_next)
    
    # 每個回合結束後，測試代理人是否能從 (0,0) 找到路徑，並記錄步數
    current_path, path_steps = find_path(Q, training_start_pos, goal_pos, R)
    steps_per_episode.append(path_steps) # 儲存此回合的步數

    # 判斷是否找到更短路徑
    if path_steps != -1 and path_steps < min_steps:
        min_steps = path_steps
        shortest_path_found = current_path
        no_improvement_count = 0 # 重置計數器，因為有改善
        print(f"回合 {episode + 1}: 找到新的最短路徑，步數為 {path_steps}。")
    elif path_steps != -1:
        no_improvement_count += 1 # 沒有改善，增加計數器
        print(f"回合 {episode + 1}: 找到路徑，步數為 {path_steps+1}。連續無改善次數: {no_improvement_count}")
    else: # 未找到路徑
        no_improvement_count += 1
        print(f"回合 {episode + 1}: 未找到路徑或代理人受困。連續無改善次數: {no_improvement_count}")
    
    # 檢查是否達到提前終止條件
    if no_improvement_count >= patience:
        print(f"\n連續 {patience} 回合最短路徑沒有改善，提前終止訓練。")
        break # 終止訓練迴圈


print("\nQ-learning 訓練完成！")
if min_steps != float('inf'):
    print(f"訓練期間從 {training_start_pos} 到目標找到的最短路徑步數為：{min_steps+1}。")
else:
    print("在整個訓練過程中未從 {training_start_pos} 找到有效路徑。")

# 繪製最短路徑圖
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(maze, cmap='gray_r') # 顯示迷宮，牆壁為黑色，路徑為白色

if shortest_path_found:
    for y, x in shortest_path_found:
        ax.plot(x, y, 'ro')  # 在路徑上的點繪製紅色圓圈
else:
    print("沒有找到有效路徑可以顯示。")

print("\n--- R (獎勵) 矩陣 ---")
print(R)
print("\n--- Q (Q 值) 矩陣 ---")
print(np.round(Q, 2)) # 將 Q 值四捨五入到小數點後兩位，使其更易讀
ax.plot(training_start_pos[1], training_start_pos[0], 'go', markersize=10, label='訓練起點')  # 標記訓練起點 (綠色圓圈)
ax.plot(goal_pos[1], goal_pos[0], 'bo', markersize=10, label='目標')  # 標記目標點 (藍色圓圈)
ax.set_title(f"Q-learning 找到的路徑 (最短步數: {min_steps if min_steps != float('inf') else 'N/A'})")
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.grid(True)
ax.legend()
plt.show()

# while True:
#     user_input = input("請輸入起點座標 (row,col) 或 'gg' 結束: ").strip().lower()

#     if user_input == 'gg':
#         print("程式結束。")
#         break

#     try:
#         parts = user_input.split(',')
#         if len(parts) != 2:
#             raise ValueError("輸入格式不正確。")
        
#         row = int(parts[0].strip())
#         col = int(parts[1].strip())
        
#         # 檢查輸入座標是否在迷宮範圍內
#         if not (0 <= row < N and 0 <= col < N):
#             print("座標超出迷宮範圍，請重新輸入。")
#             continue
        
#         # 檢查輸入座標是否是牆壁
#         if maze[row, col] == 1:
#             print("該座標是牆壁，無法作為起點。")
#             continue

#         query_start_pos = (row, col)
        
#         # 使用訓練好的 Q 矩陣來尋找路徑
#         path_for_query, steps_for_query = find_path(Q, query_start_pos, goal_pos, R)

#         if steps_for_query != -1:
#             print(f"從 {query_start_pos} 到出口需要 {steps_for_query} 步。")
#         else:
#             print("None") # 從該起點無法到達出口
            
#     except ValueError as e:
#         print(f"無效的輸入: {e}。請確保輸入為 'row,col' 格式，且為整數。")
#     except Exception as e:
#         print(f"發生未知錯誤: {e}")
