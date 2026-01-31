import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 1. 定義目標函數 (Quadratic Form)
# f(x, y) = 1.5x^2 + 2xy + 3y^2 - 2x + 8y
def f_func(x, y):
    return (3/2)*x**2 + 2*x*y + 3*y**2 - 2*x + 8*y

# 2. 定義梯度 (Gradient)
# 從目標函數微分得到
def grad_f(x, y):
    dfdx = 3*x + 2*y - 2
    dfdy = 2*x + 6*y + 8
    return np.array([dfdx, dfdy])

# 3. 定義最佳步長搜尋函數
def tausearch(tau, x, y):
    g = grad_f(x, y)
    # 計算下一個迭代點
    x_next = x - tau * g[0]
    y_next = y - tau * g[1]
    return f_func(x_next, y_next)

# 4. 初始化
x, y = 1.0, -0.2  # 初始猜測值
path = [[x, y]]   # 用於紀錄路徑

print("開始優化迭代...")

# 5. 執行最速下降法迭代
for j in range(100):
    # 搜尋當前最佳 tau
    # 使用 disp=False 隱藏 optimize.fmin 的內部輸出
    tau_opt = optimize.fmin(tausearch, 0.2, args=(x, y), disp=False)
    
    # 【關鍵修正】確保 tau 是標量數值，避免後續座標變成 array
    tau = tau_opt.item() 
    
    grad = grad_f(x, y)
    fold = f_func(x, y)
    
    # 更新座標
    x = x - tau * grad[0]
    y = y - tau * grad[1]
    f_new = f_func(x, y)
    
    # 確保存入路徑的是純數值，避免 inhomogeneous shape 錯誤
    path.append([float(x), float(y)])
    
    # 收斂判斷：前後函數值差異小於 1e-5
    if abs(f_new - fold) < 1e-5:
        print(f"迭代於第 {j+1} 次收斂。")
        print(f"最終座標: ({x:.4f}, {y:.4f}), 函數值: {f_new:.4f}")
        break

# 轉換為 numpy array 以供繪圖
path = np.array(path)

# 6. 繪製收斂路徑圖
x_range = np.linspace(-2, 4, 100)
y_range = np.linspace(-3, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f_func(X, Y)

plt.figure(figsize=(10, 8))
# 繪製等高線
cp = plt.contour(X, Y, Z, levels=40, cmap='viridis')
plt.clabel(cp, inline=True, fontsize=8)

# 繪製最速下降路徑
plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=1, label='Steepest Descent Path')
plt.plot(path[0, 0], path[0, 1], 'go', label='Start (1, -0.2)')
plt.plot(path[-1, 0], path[-1, 1], 'bo', label='Optimal Point')

plt.title('Visualization of Steepest Descent Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()