import os
import json
import numpy as np
import csv
import re
index = 4
file_name = f'instance_{index}_vi_data/instance_{index}_length_2_scaled_norm.json'
rows = 14
cols = 14
# 建立一個空的二維矩陣
my_matrix = np.zeros((14, 14))
with open(file_name) as f:
    data = json.load(f)
for key, value in data.items():
    #print(key)
    numbers = [int(num) for num in re.findall(r'\d+', key)]
    if len(numbers) ==2:
        #print(numbers[0],numbers[1])
        my_matrix[numbers[0]][numbers[1]] = value


# 下載路徑
csv_file = f"./instance_{index}_vi_data/instance_{index}_length_2_residual.csv"

# 矩陣寫入csv
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(my_matrix)

print("CSV 文件保存成功！")

