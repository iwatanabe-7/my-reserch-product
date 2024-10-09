import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize

# パペットモデルの元の座標
original_points = {
    'left_shoulder': np.array([-17.5, -50.0, 1]),
    'right_shoulder': np.array([17.5, -50.0, 1]),
    'left_hip': np.array([-11.5, 0.0, 1]),
    'right_hip': np.array([11.5, 0.0, 1])
}

# 目標座標
target_points = {
    'left_shoulder': np.array([552.4551391601562, 213.7822723388672, 0]),
    'right_shoulder': np.array([343.5639343261719, 208.13656616210938, 0]),
    'left_hip': np.array([510.1123352050781, 501.71337890625, 0]),
    'right_hip': np.array([391.5524597167969, 504.5362548828125, 0])
}

# 平行移動と回転を同時に行う関数
def transform_points(points, params):
    tx, ty = params  # 平行移動量
    transformed_points = [point + np.array([tx, ty, 0]) for point in points]  # 平行移動
    return np.array(transformed_points)

# 目的関数 (最適化したい内容)
def objective_function(params, source_points, target_points):
    transformed_points = transform_points(source_points, params)  # 各点を変換
    distances = np.linalg.norm(transformed_points - target_points, axis=1)  # 各点の距離を計算
    return np.sum(distances)  # すべての点の距離の合計を返す

# 元の四つの座標をリストに変換
source_points = np.array(list(original_points.values()))

# 目標の座標をリストに変換 (Z座標は0のまま)
target_points_array = np.array(list(target_points.values()))

# 拡大係数を設定 (6倍に拡大)
scaling_factor = 5.0  # 拡大係数
scaled_source_points = source_points * scaling_factor  # 拡大した点を計算

# 最適化実行 (初期パラメータ: 平行移動(0, 0))
initial_params = [0, 0]  # [tx, ty]
result = minimize(objective_function, initial_params, args=(scaled_source_points, target_points_array))

# 最適化結果を使って点を変換
final_transformed_points = transform_points(scaled_source_points, result.x)

# 描画
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# パペットモデルの元の点を描画 (青色の台形)
original_trapezoid_vertices = np.array([
    source_points[0],  # 左肩
    source_points[1],  # 右肩
    source_points[3],  # 右足
    source_points[2],  # 左足
    source_points[0]   # 戻るためのポイント
])
original_trapezoid_patch = Poly3DCollection([original_trapezoid_vertices], facecolors='blue', linewidths=1, edgecolors='black', alpha=0.5)
ax.add_collection3d(original_trapezoid_patch)

# 最適化結果を使った点を描画 (緑色の台形)
transformed_trapezoid_vertices = np.array([
    final_transformed_points[0],  # 左肩
    final_transformed_points[1],  # 右肩
    final_transformed_points[3],  # 右足
    final_transformed_points[2],  # 左足
    final_transformed_points[0]    # 戻るためのポイント
])

transformed_trapezoid_patch = Poly3DCollection([transformed_trapezoid_vertices], facecolors='green', linewidths=1, edgecolors='black', alpha=0.5)
ax.add_collection3d(transformed_trapezoid_patch)

# 目標の点を描画 (赤色)
for name, point in target_points.items():
    ax.scatter(*point, color='red', label=f'Target {name}', s=100)

# 目標の台形の外枠を描画 (赤色)
target_trapezoid_vertices = np.array([
    target_points['left_shoulder'],
    target_points['right_shoulder'],
    target_points['right_hip'],
    target_points['left_hip'],
    target_points['left_shoulder']  # 戻るためのポイント
])
ax.plot(*zip(*target_trapezoid_vertices), color='red', alpha=0.5)  # 目標の台形の外枠を結ぶ

# ラベル設定
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Original Trapezoid (Blue) and Transformed Trapezoid (Green)')
ax.view_init(elev=90, azim=90)  # Z軸から見た視点に設定

# 軸の範囲を調整して全体を見やすくする
ax.set_xlim([-600, 600])
ax.set_ylim([-600, 600])
ax.set_zlim([-100, 100])

plt.legend()
plt.show()

# 結果を表示
print("最適化結果: 平行移動 (tx, ty) =", result.x)
