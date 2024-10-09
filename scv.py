import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import svd


def read_head_section(file_path):
    head_data = []
    # stop_keyword = 'left foot'
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip the first line (header)
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True  # Skip header line
                continue
            
            # Check if the row contains the stop keyword
            # if row and len(row) > 0 and row[0] == stop_keyword:
            #     print("Encountered stop keyword; stopping read.")
            #     break
            
            # Check if the row is not empty and has the expected number of columns
            if row and len(row) >= 2:
                try:
                    # Convert the values to floats and append to the list
                    x = float(row[0])
                    y = float(row[1])
                    head_data.append([x, y])
                except ValueError:
                    print(f"Skipping invalid row: {row}")
            else:
                print(f"Skipping malformed row: {row}")
    
    return head_data

file_path = 'data/result/light-friend-t-20m.csv'

array = read_head_section(file_path)

head_x = [float(row[0]) for row in array]
head_y = [float(row[1]) for row in array]

# numpy配列に変換
# data = np.array(head_y)

# 元のデータ (head_y) をプロット
plt.plot(head_x, head_y, marker='o', linestyle='-', color='b', label='Head')
# plt.plot(range(len(array)), array, marker='o', linestyle='-', color='b', label='Original Head')
# plt.plot(range(len(head_y)), head_y, marker='o', linestyle='-', color='b', label='Original Head')

# 修正後のデータ (data) をプロット
# plt.plot(range(len(filtered_data)), filtered_data, marker='x', linestyle='-', color='r', label='Corrected Head')

# グラフのタイトルとラベルの設定
plt.title('Head Data (Original vs Corrected)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.gca().invert_yaxis()
plt.grid(True)

plt.legend()


plt.show()