import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# カメラパラメータ
fx = 2.80534073
fy = 2.80228165
cx = 2.02546251
cy = 1.49613246

# パペットモデルの座標
u = -17.5
v = -50.0

puppet_uv = [
    [-17.5, -50.0],
    [17.5, -50.0],
    [-11.5, 0.0],
    [11.5, 0.0]
]

# mmpose キーポイント

mmpose_xyz = [
    [552.4551391601562, 213.7822723388672, 1970],
    [343.5639343261719, 208.13656616210938, 1970],
    [510.1123352050781, 501.71337890625, 1970],
    [391.5524597167969, 504.5362548828125, 1970]
]

A = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# 回転行列・平行移動させて2次元化
def Rotation(params,s):
    x_angle,y_angle,z_angle,t1,t2,t3 = params
    x = math.radians(x_angle)
    y = math.radians(y_angle)
    z = math.radians(z_angle)
    rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0,  math.sin(x),  math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y),  0,  math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0,  0,  1]])
    # rz = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0,  0,  1]])
    t = np.array([t1, t2, t3])
    return (rx @ ry @ rz @ s + t) @ A

# x_angle,y_angle,z_angle,t1,t2,t3を最適化
def optimization(params):
    total_error = 0
    for idx in range(len(puppet_uv)):
        u, v = puppet_uv[idx]
        s = np.array([u, v, 1])
        i = np.array(mmpose_xyz[idx])
        total_error += np.linalg.norm(Rotation(params,s) - i)
    
    return total_error

init_params = [1,1,1,1,1,1]
result = optimize.fmin(optimization,init_params)

print(f"最適化された値：{result}")

rotation_result = []
for i in range(len(puppet_uv)):
    u, v = puppet_uv[i]
    s = np.array([u, v, 0])
    rotation_result.append(Rotation(result,s))
    print(f"{(i+1)}:{Rotation(result,s)}")


# puppet_uvとmmpose_xyzと最適化したものを描写
puppet_uv = [
    [-17.5, -50.0, 1],
    [17.5, -50.0, 1],
    [-11.5, 0.0, 1],
    [11.5, 0.0, 1]
]

puppet_uv = np.array(puppet_uv)
mmpose_xyz = np.array(mmpose_xyz)
rotation_result = np.array(rotation_result)

# プロット
fig = plt.figure()

# 3Dプロットの作成
ax = fig.add_subplot(111, projection='3d')

# puppet_uvのプロット（青）
verts = [[
    puppet_uv[0], puppet_uv[1], puppet_uv[3], puppet_uv[2]
]]
ax.add_collection3d(Poly3DCollection(verts, color='blue', alpha=0.5))
ax.scatter(puppet_uv[:, 0], puppet_uv[:, 1], puppet_uv[:, 2], c='blue', label='Puppet')

# mmpose_xyzのプロット（赤）
verts = [[
    mmpose_xyz[0], mmpose_xyz[1], mmpose_xyz[3], mmpose_xyz[2]
]]
ax.add_collection3d(Poly3DCollection(verts, color='red', alpha=0.5))
ax.scatter(mmpose_xyz[:, 0], mmpose_xyz[:, 1], mmpose_xyz[:, 2], c='red', label='mmpose')

# 最適化結果のプロット（緑）
verts = [[
   rotation_result[0], rotation_result[1], rotation_result[3], rotation_result[2]
]]
ax.add_collection3d(Poly3DCollection(verts, color='green', alpha=0.5))
ax.scatter(rotation_result[:, 0], rotation_result[:, 1], rotation_result[:, 2], c='green', label='result')

# 軸ラベルの設定
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# タイトルと凡例
ax.set_title('Puppet UV, mmpose Result')
ax.view_init(elev=-90, azim=-90)  # Z軸から見た視点に設定
ax.legend()

# プロット表示
plt.show()
