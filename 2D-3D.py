import argparse
import math
import numpy as np
import itertools

from scipy import optimize
import matplotlib.pyplot as plt

# コマンドライン引数を設定
parser = argparse.ArgumentParser(description="Optimization Result Logger")
parser.add_argument("-o", "--output", type=str, help="Output file name for results")
args = parser.parse_args()

# カメラパラメータ(px)
fx = 2805.78
fy = 2802.63
cx = 2026.51
cy = 1496.88

# 実際の体の長さ(cm)
shoulder_len = 38
waist_len = 24
shoulder_to_waist_len = 50
# shoulder_len = 35
# waist_len = 23
# shoulder_to_waist_len = 50
# shoulder_to_waist_len = 25


shoulder_half_len = shoulder_len / 2
waist_half_len = waist_len / 2

# キーポイントの値
key_point = [[479.45, -1219.49],[672.51, -1216.56],[523.33, -1485.67],[634.48, -1485.67]]

# パペットモデルの座標(cm)
puppet_uv = np.array([
    [-shoulder_half_len, shoulder_to_waist_len, 0],
    [shoulder_half_len, shoulder_to_waist_len, 0],
    [-waist_half_len, 0.0, 0],
    [waist_half_len, 0.0, 0]
])

# mmpose キーポイント(px)
mmpose_xyz = np.array(key_point)

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
    t = np.array([t1, t2, t3])

    r = rx @ ry @ rz
    s_rotated = r @ s.T
    s_translated = s_rotated.T + t
    s_result = A @ s_translated.T
    scale = s_result[2]
    s_result /= scale
    return s_result[:2]

# x_angle,y_angle,z_angle,t1,t2,t3を最適化
def optimization(params):
    total_error = 0
    for idx in range(len(puppet_uv)):
        u, v , p= puppet_uv[idx]
        s = np.array([u, v, p])
        i = np.array(mmpose_xyz[idx])
        projected_points = Rotation(params, s)
        total_error += np.linalg.norm(projected_points - i)
    return total_error

# angle_range = np.arange(-10, 11, 5)  # 回転角の範囲（-10度から10度まで、5度刻み）
# translation_range = np.arange(-50, 51, 25)  # 平行移動の範囲（-50cmから50cmまで、25cm刻み）
# param_combinations = list(itertools.product(angle_range, repeat=3)) + \
#                      list(itertools.product(translation_range, repeat=3))
# # 全組み合わせの最適化を実行
# for angles in param_combinations:
#     for translations in itertools.product(translation_range, repeat=3):
#         init_params = angles + translations
#         result = optimize.minimize(optimization, init_params, method='L-BFGS-B')
        
#         print(f"初期パラメータ: {init_params}")
#         print(f"最適化結果: {result.x}")
#         print(f"目的関数の値: {result.fun}\n")

# init_params = [np.random.uniform(-30, 30), np.random.uniform(-30, 30), np.random.uniform(-30, 30),
#                np.random.uniform(-300, 0), np.random.uniform(-500, -400), np.random.uniform(400, 500)]
init_params = [0.5, 0.5, 0.5, 0, 0, 300]
# init_params = [0.1, 0.1, 0.1, 0, 0, 1000]
result = optimize.minimize(optimization, init_params, method='L-BFGS-B')

# 各座標に対して最適化された回転・平行移動を適用
optimized_projection = []
for idx in range(len(puppet_uv)):
    u, v, p = puppet_uv[idx]
    s = np.array([u, v, p])
    optimized_point = Rotation(result.x, s)
    optimized_projection.append(optimized_point)

# カメラの位置を取得
camera_position = result.x[5]
camera_position /= 100

#出力を保存
if args.output is not None:
    with open(args.output, "a", encoding="utf-8") as file:
        # カメラパラメータ
        file.write(f"fx: {fx}\n")
        file.write(f"fy: {fy}\n")
        file.write(f"cx: {cx}\n")
        file.write(f"cy: {cy}\n")
        file.write("\n")

        # 実際の体の長さ
        file.write(f"shoulder_len: {shoulder_len} cm\n")
        file.write(f"waist_len: {waist_len} cm\n")
        file.write(f"shoulder_to_waist_len: {shoulder_to_waist_len} cm\n")
        file.write("\n")

        # キーポイント
        file.write("key_point:\n")
        for point in key_point:
            file.write(f"{point}\n")
        file.write("\n")

        # パペットモデルの座標
        file.write("puppet_uv:\n")
        for puppet_point in puppet_uv:
            file.write(f"{puppet_point}\n")
        file.write("\n")
        file.write("\n--- 最適化結果 ---\n")
        file.write(f"完了した理由：{result.message}\n")
        file.write(f"結果：{result.success}\n")
        file.write(f"最適化の状態(0が成功): {result.status}\n")
        file.write(f"目的関数の値：{result.fun}\n")
        file.write(f"最適化されたパラメータ： {result.x}\n")
        file.write(f"反復回数： {result.nit}\n")
        file.write(f"勾配： {result.jac}\n")
        file.write(f"目的関数が評価された回数： {result.nfev}\n")
        file.write(f"勾配が評価された回数： {result.njev}\n")
        file.write(f"曲率（逆ヘッセ行列）: {result.hess_inv}\n")
        file.write(f"カメラと被写体の距離: {camera_position:.2f} m\n")
        file.write("\n\n")
    print("出力を書き込みました")
else:
    print(f"完了した理由：{result.message}")
    print(f"結果：{result.success}")
    print(f"最適化の状態(0が成功):{result.status}")
    print(f"目的関数の値：{result.fun}")
    print(f"最適化されたパラメータ：{result.x}")
    print(f"反復回数：{result.nit}")
    print(f"勾配：{result.jac}")
    print(f"目的関数が評価された回数：{result.nfev}")
    print(f"勾配が評価された回数：{result.njev}")
    print(f"曲率：{result.hess_inv}")
    print(f"カメラと被写体の距離: {camera_position:.2f} m")

# puppet_uv(cm)とmmpose_xyzと最適化したものを描写(描写のために座標を入れ替え)
numbers = [0,1,3,2]
puppet_uv = np.array([puppet_uv[i] for i in numbers])
mmpose_xyz = np.array([mmpose_xyz[i] for i in numbers])
rotation_result = np.array([np.round(optimized_projection[i], 2) for i in numbers])

print(mmpose_xyz)
print(rotation_result)
# 平均二乗誤差を計算
mse = np.mean((mmpose_xyz - rotation_result) ** 2)
print("Mean Squared Error (MSE):", mse)

plt.figure(figsize=(10, 6))

plt.fill(*zip(*puppet_uv), color='blue', alpha=0.5, label='Puppet UV')
plt.fill(*zip(*mmpose_xyz), color='red', alpha=0.5, label='mmpose XYZ')
plt.fill(*zip(*rotation_result), color='green', alpha=0.5, label='Optimized Result')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Puppet UV, mmpose, Result')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
