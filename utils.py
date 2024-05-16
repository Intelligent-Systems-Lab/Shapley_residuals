import os
import json
import re
def find_subset(index,feature_num=14):
    
    '''To find the subset that will decrease the scaled norm with observerble difference'''
    
    file_name = f'./instance_{index}_vi_data/instance_{index}_allsubsets_scaled_norm.json'
   
    with open(file_name) as f:
        data = json.load(f)
    A = []
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