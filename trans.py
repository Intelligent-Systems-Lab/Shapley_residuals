import os
import json
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
from itertools import combinations
from feature_name import feature_name
from partial_gradient_square import partial_gradient_square
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

#######
#######
'''                 For subset              '''

def trans_json_to_dictinary(json_file_name):

    with open(json_file_name) as f:
        data = json.load(f)
    return data
    
    

def create_different_len_subset_list(data,feature_num=14,non_consider_feature=[]):
    ''' input dictionary and output list'''
    flag = True
    different_len_subset_list = [{} for _ in range(feature_num)]        #dictionaries for different length
    for key, value in data.items():
        flag = True
        numbers = [int(num) for num in re.findall(r'\d+', key)] 
        for feature in non_consider_feature:
            if feature in numbers:
                flag = False
        if flag:
            different_len_subset_list [len(numbers)-1][key] = value         
    return different_len_subset_list

def instance_data_to_subset_list(index,feature_num=14):
    '''input index and out put list'''
    file_name = f'./instance_{index}_vi_data/instance_{index}_allsubsets_scaled_norm.json'
    dict = trans_json_to_dictinary(file_name)
    data = create_different_len_subset_list(dict,feature_num=feature_num,non_consider_feature=[])
    return data

def generate_all_subsets(num):                     #
    num_set = [i for i in range(num)]
    all_subsets = [np.array(s) for r in range(num+1) for s in combinations(num_set, r) ]
    # Create a hash table to store the index of each subset
    return all_subsets

def show_fig_single_feature(idx_list,feature_num=14):

    ''' input instances_index_list and show scattered graph of single feature from different instances'''

    stored_list = [[] for _ in range(feature_num)]
    for idx in idx_list:
        data = instance_data_to_subset_list(idx)
        # store each feature scaled norm
        for i  in range(feature_num):
            values = list(data[0].values())
            stored_list[i].append(values[i])
        #print(stored_list)
    colors = plt.cm.tab20.colors
    for i, group in enumerate(stored_list):
        x_values = [feature_name[i]] * len(group)  # 將 x 軸值調整為同一組的索引值，讓點排列在同一條直線上
        plt.scatter(x_values, group, color=colors[i % len(colors)])

    
    plt.xlabel('Feature')
    plt.ylabel('Scaled norm')
    plt.title('Single_feature_scaled_norm')
    plt.xticks(rotation=60, ha='right')  # 將 x 軸標籤旋轉 45 度，並對齊右邊
    plt.tight_layout()  # 調整圖形布局以避免標籤重疊
    plt.show()

def show_different_subset_graph(index,subset_len=2,feature_num=14):
    
    data = instance_data_to_subset_list(index)
    subset_idx = subset_len - 1
    values = list(data[subset_idx].values())
    plt.ylim(0,1.2)
    plt.scatter(range(len(values)), values)
    plt.title(f'length{subset_len}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

### todo
'''def relative_decreasing_subset(index_list=[],partial_gradient_square=[],subset_len=2,feature_num=14,max_subset_len=6):
    for index in index_list:
        file_name = f'./instance_{index}_vi_data/instance_{index}_allsubsets_scaled_norm.json'
        data = instance_data_to_subset_list(index)
        dict = trans_json_to_dictinary(file_name)
        filtered_data = []
        for i in range(max_subset_len):
            filtered_dict = {}
            for key,value in data[i].items():
                key_list = [int(num) for num in re.findall(r'\d+', key)] 
                count =0 
                flag = True
                partial_gradient_square_sum = sum(partial_gradient_square[index][j] for j in key_list)
                if i!=0:
                    subsets= [np.array(subset) for subset in combinations(key_list, i)]
                    #print(key_list)
                    #print(subsets)
                    for subset in subsets:
                        partial_gradient_square_subset_sum = sum(partial_gradient_square[index][k] for k in subset)
                        weight = (partial_gradient_square_subset_sum/partial_gradient_square_sum)**0.5 
                        #print(weight)
                        if value > dict[str(subset)]*1:
                            flag = False
                            #filtered_dict[key] =  value
                            #print(str(subset))
                    if flag:
                        filtered_dict[key] = value
            filtered_data.append(filtered_dict)    '''