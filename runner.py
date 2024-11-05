import matplotlib.pyplot as plt
import cv2
import os
import csv
from mmpose.apis import MMPoseInferencer

# 自分のジョグ
# img_folder_path = 'data/image/my-run-light-my-jog-20m'
# output_folder_path = 'data/result/light-my-jog-20m'
# output_file = 'data/result/light-my-jog-20m.csv'

# 友達の20mの走り
# img_folder_path = 'data/image/my-run-light-friend-t-20m'
# output_folder_path = 'data/result/light-friend-t-20m'
# output_file = 'data/result/light-friend-t-20m.csv'

# 自分の競技場での走り
# img_folder_path = 'data/image/my-run-dark-10'
# output_folder_path = 'data/result/dark-10-change-constract-3-40'
# output_file = 'data/result/dark-10-change-contract-3-40.csv'

#自分の競技場での走り2
# img_folder_path = 'data/image/my-run-dark-10-3-lite'
# output_folder_path = 'data/result/dark-10-3-lite'
# output_file = 'data/result/dark-10-3-lite.csv'

# pro-jの走り
# img_folder_path = 'data/image/pro-j'
# output_folder_path = 'data/result/pro-j'
# output_file = 'data/result/pro-j.csv'

# オリンピック準決勝3組
# img_folder_path = 'data/image/olympics/olympics_semifinal_triple_throw'
# output_folder_path = 'data/result/olympics/olympics_semifinal_triple_throw'
# output_file = 'data/result//olympics_semifinal_triple_throw.csv'

img_folder_path = 'data/image/stand'
output_folder_path = 'data/result/stand'
output_file = 'data/result/stand.csv'



# Initialize the inferencer
inferencer = MMPoseInferencer('wholebody')

print("実行しています")

# Get the results
result_generator = inferencer(img_folder_path, vis_out_dir=output_folder_path, show=False)
# result_generator = inferencer(img_folder_path, show=False)
results = [result for result in result_generator]

print(results)

def view_keypoint(i,re):
    tmp = i+1
    if(tmp == 5 or tmp == 16 or tmp == 17):
        if(tmp == 5):print("頭の中心") 
        elif(tmp==16):print("左足")
        elif(tmp==17):print("右足")
        
        print(re[i])

head = []
left_leg = []
right_leg = []

for i,result in enumerate(results):
    print(i+1)
    print("枚目の画像")
    for j, re in enumerate(result['predictions'][0][0]['keypoints']):
        tmp = j+1
        if(tmp == 5 or tmp == 16 or tmp == 17):
            if(tmp == 5):
                print("頭の中心") 
                head.append(re)
            elif(tmp==16):
                print("左足")
                left_leg.append(re)
            elif(tmp==17):
                print("右足")
                right_leg.append(re)
            print(re)
    print("\n")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    writer.writerow(["head"])
    writer.writerow(["x", "y"])
    for coord in head:
        writer.writerow(coord)
    
    writer.writerow([])
    writer.writerow(["left foot"])
    writer.writerow(["x", "y"])
    for coord in left_leg:
        writer.writerow(coord)
    
    writer.writerow([])
    writer.writerow(["right foot"])
    writer.writerow(["x", "y"])
    for coord in right_leg:
        writer.writerow(coord)

print(f"Coordinates saved to {output_file}")

def extract_coordinates(keypoints):
    x = [kp[0] for kp in keypoints]
    y = [kp[1] for kp in keypoints]
    return x, y

head_x, head_y = extract_coordinates(head)
left_foot_x, left_foot_y = extract_coordinates(left_leg)
right_foot_x, right_foot_y = extract_coordinates(right_leg)

plt.figure(figsize=(10, 8))
# plt.plot(head_x, head_y, marker='o', linestyle='-', color='b', label='Head')
plt.plot(left_foot_x, left_foot_y, marker='o', linestyle='-', color='g', label='Left Foot')
plt.plot(right_foot_x, right_foot_y, marker='o', linestyle='-', color='r', label='Right Foot')

plt.title('Keypoints Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.show()
