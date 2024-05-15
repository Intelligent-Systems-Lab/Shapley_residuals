import os
import json
import numpy as np 
from itertools import combinations
from tqdm import tqdm
from trans import length_2_subset_dependency
partial_gradient_norm  = [323.55596783123735, 16.64309301545677, 20.664327935240934, 0.7640419085988084, 8.322414016557218, 36.0900489996741, 2.3600512407406358, 2.023149132555818, 0.3237797476085129, 1.722481500606519, 0.10121379279226961, 0.00015068922881678048, 0.9322766973094891, 6.387281114990409]
index = 5
folder_path = f"./instance_{index}_vi_data"
file_extension = 'res.json'
params_list = ['X_-TACT_TIME_mean', 'X_-CONVEYOR_SPEED_mean', 'PUMP_high', 'PUMP_low', 'CLN1_over-etching-ratio', 'CLN1_EPT_time', 'clean_count', 'EPT_clean_count_ratio', 'NH3_TREAT_-RF_FREQ-max', 'NH3_TREAT_-RF_FREQ-range', 'NH3_TREAT_-RF_FREQ-mean', 'NP_3_-MFC_VOL_SIH4-range', 'VENT_high', 'VENT_low', 'DFT_CNT']
def generate_all_subsets(num):                     #
    num_set = [i for i in range(num)]
    all_subsets = [np.array(s) for r in range(num+1) for s in combinations(num_set, r) ]
    # Create a hash table to store the index of each subset
    return all_subsets
if not os.path.exists(folder_path):
    print('資料夾不存在')
    exit()

file_names = os.listdir(folder_path)
json_files = []
for feature in params_list:
    for file_name in file_names:
        if (feature+'_res') in file_name:
            json_files.append(file_name)         #提取需要的資料名稱

res_vector_data = []
for json_file in json_files:
   file_path = os.path.join(folder_path,json_file)
   with open(file_path) as f:
       data = json.load(f)
       res_vector_data.append(data)               #取得所有residual_cube_edge的值

AUO_coalitions = generate_all_subsets(14)
length_2_coalitions = [coalition for coalition in AUO_coalitions if len(coalition)<=2]
residual_norm_of_each_subset = {}
for coalition in tqdm(length_2_coalitions ):        #對所有subsets
    if len(coalition)!=0:
        residual_vector_sum = {}
        partial_gradient_subset_norm = 0
        turn  = 0 
        for feature in coalition:           #對subset裡的每一個特徵
            partial_gradient_subset_norm += partial_gradient_norm[feature]          #加上subset中每個特徵的partial gradient norm
            if turn==0:
                for key in res_vector_data[feature]:            #對每個特徵的residual vector的每一條邊
                    residual_vector_sum[key] = res_vector_data[feature][key]
                turn = 1
            else:
                for key in res_vector_data[feature]:           #對每個特徵的residual vector的每一條邊
                    residual_vector_sum[key] += res_vector_data[feature][key]             #把所有的residual_vector做相加
        residual_subset_norm = sum(edge**2 for edge in residual_vector_sum.values())
    
        residual_norm_of_each_subset[str(coalition)] = (residual_subset_norm/partial_gradient_subset_norm)**0.5

print(residual_norm_of_each_subset)
saved_file_name = f'./instance_{index}_vi_data/instance_{index}_length_2_scaled_norm.json'
with open(saved_file_name, 'w') as json_file:
    json.dump(residual_norm_of_each_subset, json_file)
    
length_2_subset_dependency(index=index)
#print(res_vector_data)
#for json_file in json_files:
    #file_path = os.path