import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.image as mpimg

# 上底と下底の長さ
bottom_length = 23  # 腰の長さ
top_length = 35     # 肩の長さ
height = 50         # 肩から腰の高さ
z_shift = 100       # 台形をZ軸方向に持ち上げる
leg_length = 37 + 41  # 足の長さ
hand_length = 27 + 25  # 手の長さ
neck_length = 16     # 首の長さ

# 台形の平行移動量を計算 (左上を原点に)
x_shift = 0
y_shift = 25
z_shift = 0

# 台形の頂点座標 (x, y, z) - 左上原点にシフト
# 台形の頂点座標 (Y方向を反転)
trapezoid = np.array([[-bottom_length / 2 + x_shift, height / 2 - y_shift, z_shift],  # 下底の左端（Y反転）
                      [bottom_length / 2 + x_shift, height / 2 - y_shift, z_shift],   # 下底の右端（Y反転）
                      [top_length / 2 + x_shift, -height / 2 - y_shift, z_shift],      # 上底の右端
                      [-top_length / 2 + x_shift, -height / 2 - y_shift, z_shift],     # 上底の左端
                      [-bottom_length / 2 + x_shift, height / 2 - y_shift, z_shift]])  # 下底の左端に戻る

# 足の座標 (x, y, z) - 左上原点にシフト
legs = np.array([[-bottom_length / 2 + x_shift, -height / 2 + y_shift, z_shift],         # 左足の上端
                 [-bottom_length / 2 + x_shift, -height / 2 - leg_length + y_shift, z_shift],  # 左足の下端
                 [bottom_length / 2 + x_shift, -height / 2 + y_shift, z_shift],          # 右足の上端
                 [bottom_length / 2 + x_shift, -height / 2 - leg_length + y_shift, z_shift]])  # 右足の下端

# 手の座標 (x, y, z) - 左上原点にシフト
hands = np.array([[-top_length / 2 - hand_length + x_shift, height / 2 + y_shift, z_shift],  # 左手の終点
                  [-top_length / 2 + x_shift, height / 2 + y_shift, z_shift],                # 左手の開始点
                  [top_length / 2 + x_shift, height / 2 + y_shift, z_shift],                # 右手の開始点
                  [top_length / 2 + hand_length + x_shift, height / 2 + y_shift, z_shift]])  # 右手の終点

# 首の座標 (x, y, z) - 左上原点にシフト
neck_center = np.array([x_shift, height / 2 + y_shift, z_shift])  # 台形の上部中央
neck = np.array([[neck_center[0], neck_center[1], neck_center[2]],  # 首の下端
                 [neck_center[0], neck_center[1] + neck_length, neck_center[2]]])  # 首の上端

# 描画
fig = plt.figure(figsize=(12, 8))

# 3Dプロット
ax = fig.add_subplot(111, projection='3d')

# 台形を描画
trapezoid_patch = Poly3DCollection([trapezoid], facecolors='lightblue', linewidths=1, edgecolors='black', alpha=0.5)
ax.add_collection3d(trapezoid_patch)

# 足を描画 (X軸とY軸の符号を逆にする)
ax.plot([-legs[0, 0], -legs[1, 0]], [-legs[0, 1], -legs[1, 1]], [legs[0, 2], legs[1, 2]], color='r')  # 左足
ax.plot([-legs[2, 0], -legs[3, 0]], [-legs[2, 1], -legs[3, 1]], [legs[2, 2], legs[3, 2]], color='r')  # 右足

# 手を描画 (X軸とY軸の符号を逆にする)
ax.plot([-hands[0, 0], -hands[1, 0]], [-hands[0, 1], -hands[1, 1]], [hands[0, 2], hands[1, 2]], color='g')  # 左手
ax.plot([-hands[2, 0], -hands[3, 0]], [-hands[2, 1], -hands[3, 1]], [hands[2, 2], hands[3, 2]], color='g')  # 右手

# 首を描画 (X軸とY軸の符号を逆にする)
ax.plot(-neck[:, 0], -neck[:, 1], neck[:, 2], color='k', linewidth=2, label='Neck')  # 首

image_path = './data/result/stand/front.jpg'  # 画像ファイルのパスを指定
img = mpimg.imread(image_path)

img_height, img_width = img.shape[:2]

# X = np.linspace(0, top_length / 2 + hand_length + x_shift, img_width)
# # Y軸を反転し、上下逆に
# Y = np.linspace(0, height / 2 + leg_length + y_shift, img_height) + leg_length + y_shift
# X, Y = np.meshgrid(X, Y)
# Z = -np.full_like(X, z_shift)

X = np.linspace(0, top_length / 2 + hand_length + x_shift, img_width)
# Y軸を反転し、上下逆に
Y = np.linspace(height / 2 + leg_length + y_shift, 0, img_height) + leg_length + y_shift

X, Y = np.meshgrid(-X, Y)
Z = np.zeros_like(X)  # Z軸をすべて0に設定


img_normalized = img / 255.0

# 軸設定
# ax.set_xlim(0, top_length / 2 + hand_length + x_shift)
# ax.set_ylim(0, height / 2 + leg_length + y_shift)
# ax.set_zlim(z_shift - 10, z_shift)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D View with Image on XY Plane (Top-Left Origin)')

# 視点設定
ax.view_init(elev=90, azim=-270)  # XY平面を見下ろす視点

plt.show()


