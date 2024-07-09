import numpy as np
import pickle
from trans import generate_all_subsets
from tqdm import tqdm


def find_adjacent(para_num):
    all_subsets = generate_all_subsets(para_num)
    vertex_adjacent = [[] for _ in range(len(all_subsets))]
    partial_edges = [[] for i in range(para_num)]
    #print(len(vertex_adjacent))
    for i in tqdm(range(len(all_subsets))):
        adjacent_count = len(vertex_adjacent[i])    #已經有紀錄相鄰的點數
        for j in range(i+1,len(all_subsets)):
            if len(all_subsets[j]) - len(all_subsets[i]) == 1 and adjacent_count<para_num:         #如果兩個子集合長度差為1
                if len(np.intersect1d(all_subsets[i], all_subsets[j])) == len(all_subsets[i]):      #如果交集恰好為all_subsets[i]
                    vertex_adjacent[i].append(j)       #紀錄j的編號
                    vertex_adjacent[j].append(i)        
                    adjacent_count += 1
                    #####
                    diff_f = np.setdiff1d(all_subsets[j], all_subsets[i]).item()   #前面array減掉後面array
                    partial_edges[diff_f].append([i,j])
            elif len(all_subsets[i]) == len(all_subsets[j]):
                continue
            else:
                break
        
    return vertex_adjacent, partial_edges





####這個部分需要改成字典操作
def mapping_adjacent(para_num,idx_list):
    map = []
    all_subsets = generate_all_subsets(para_num)
    for idx in idx_list:
        map.append(all_subsets[idx])
    return map

for num in range(1,15):
    adjacent_list,partial_edges_list = find_adjacent(num)
    # 保存列表到文件
    with open(f'vertex_adjacent_{num}.pkl', 'wb') as f:
        pickle.dump(adjacent_list, f)
        
    with open(f'partial_edges_{num}.pkl', 'wb') as f:
        pickle.dump(partial_edges_list, f)
    # 讀取文件
    #with open(f'vertex_adjacent_{num}.pkl', 'rb') as f:
        #loaded_list = pickle.load(f)

    #print(loaded_list)