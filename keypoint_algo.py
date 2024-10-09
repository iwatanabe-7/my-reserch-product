import numpy as np
from scipy.optimize import fmin

# 回転行列を作成 (X, Y, Z軸に対する回転)
def rotation_matrix(params):
    theta_x, theta_y, theta_z = params  # 各軸の回転角度
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

# 平行移動と回転を同時に行う関数
def transform_point(point, params):
    tx, ty, tz, theta_x, theta_y, theta_z = params  # 平行移動量と回転角度
    R = rotation_matrix([theta_x, theta_y, theta_z])  # 回転行列
    transformed_point = R @ point + np.array([tx, ty, tz])  # 回転+平行移動
    return transformed_point

# 目的関数 (最適化したい内容)
def objective_function(params, source_points, target_points):
    transformed_points = [transform_point(point, params) for point in source_points]  # 各点を変換
    distances = [np.linalg.norm(transformed - target) for transformed, target in zip(transformed_points, target_points)]
    return np.sum(distances)  # すべての点の距離の合計を返す

# 元の四つの座標 (左肩、右肩、左腰、右腰)
source_points = np.array([
    [-17.5, -50.0, 0],  # 左肩
    [17.5, -50.0, 0],   # 右肩
    [-11.5, 0.0, 0],    # 左腰
    [11.5, 0.0, 0]      # 右腰
])

# 目標の座標 (新しい座標)
target_points = np.array([
    [552.4551391601562, 213.7822723388672, 0],  # 左肩
    [343.5639343261719, 208.13656616210938, 0],  # 右肩
    [510.1123352050781, 501.71337890625, 0],    # 左腰
    [391.5524597167969, 504.5362548828125, 0]    # 右腰
])

# 最適化実行 (初期パラメータ: 平行移動(0, 0, 0)と回転角度(0, 0, 0))
initial_params = [0, 0, 0, 0, 0, 0]  # [tx, ty, tz, theta_x, theta_y, theta_z]
result = fmin(objective_function, initial_params, args=(source_points, target_points))

# 結果を表示
print("最適化結果: 平行移動 (tx, ty, tz) =", result[:3], "回転角度 (theta_x, theta_y, theta_z) =", result[3:])
