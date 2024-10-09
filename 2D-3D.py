import math
import numpy as np
from scipy import optimize

# カメラパラメータ
fx = 2.80534073
fy = 2.80228165
cx = 2.02546251
cy = 1.49613246

# パペットモデルの座標
u = -17.5
v = -50.0

# mmpose キーポイント
mp_x = 552.4551391601562
mp_y = 213.7822723388672
mp_z = 100

A = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

s = np.array([u, v, 0])

i = np.array([mp_x, mp_y, mp_z])

# 回転行列・平行移動させて2次元化
def Rotation(params):
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
    t = np.array([t1, t2, t3])
    return (rx @ ry @ rz @ s + t) @ A

# x_angle,y_angle,z_angle,t1,t2,t3を最適化
def optimization(params):
    return np.linalg.norm(Rotation(params) - i)

init_params = [1,1,1,1,1,1]
result = optimize.fmin(optimization,init_params)

print(f"最適化された値：{result}")
print(f"移動した点の座標：{Rotation(result)}")