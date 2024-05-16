import os
import json
import numpy as np
import csv
import re

def length_2_subset_dependency(index=4,feature_num=14):
    '''transform length 2 subset scaled norm to matrix and csv file'''
    
    file_name = f'instance_{index}_vi_data/instance_{index}_length_2_scaled_norm.json'
    rows = feature_num
    cols = feature_num
    # 建立一個空的二維矩陣
    my_matrix = np.zeros((rows, cols))
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

def find_a_subset(index,feaure=14):
    
    '''To find the subset that will decrease the scaled norm with observerble difference'''
    
    file_name = f'./instance_{index}_vi_data/instance_{index}_allsubsets_scaled_norm.json'
    feature_num =14
    with open(file_name) as f:
        data = json.load(f)
    A = []      #saled norm of each feature
    for key, value in data.items():
            #print(key)
            numbers = [int(num) for num in re.findall(r'\d+', key)]
            if  len(numbers)==1:
                #print(numbers[0],numbers[1])
                A.append(value)

    for i in range(0,feature_num):
        min_value = 1
        for key, value in data.items():
            #print(key)
            numbers = [int(num) for num in re.findall(r'\d+', key)]
            B = [A[j]-0.01 for j in numbers]
            #print(B)
            if  numbers[0]==i and value<(min_value-0.02) and value<min(B):
                #print(numbers[0],numbers[1])
                min_value = value
                print('feature_subset:',numbers,'scaled norm:',value)
    

def find_global_min(index,subset_len,non_consider_feature=[]):
    
    '''find the global min of subset_len, you can put non conisder features in the list'''
    
    file_name = f'./instance_{index}_vi_data/instance_{index}_allsubsets_scaled_norm.json'
    subset_dic = {}
    with open(file_name) as f:
        data = json.load(f)
    A = []      #scaled norm of each feature
    for key, value in data.items():
            #print(key)
            numbers = [int(num) for num in re.findall(r'\d+', key)]
            if  len(numbers)==1:
                #print(numbers[0],numbers[1])
                A.append(value)
            
            B = [A[j] for j in numbers]             #find scaled norm of each
            flag = True
            for feature in non_consider_feature:
                if feature in numbers:
                    flag = False
                    break
    
            if len(numbers)==subset_len and flag:
            #print(key)
                subset_dic[key] = value
            else:
                continue
    
    sorted_subset_dic = dict(sorted(subset_dic.items(), key=lambda item: item[1]))
    
    keys_smallest_10_percent = list(sorted_subset_dic.items())[:int(len(sorted_subset_dic)*0.1)]
    print(A)
    for item in keys_smallest_10_percent:
        print(item)
    