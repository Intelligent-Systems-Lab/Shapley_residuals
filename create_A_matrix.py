import scipy.sparse as sps
import numpy as np
import pickle

def create_A_matrix(para_num):
    v_num = 2**para_num
    with open(f'vertex_adjacent/vertex_adjacent_{para_num}.pkl', 'rb') as f:
            adjacent_s = pickle.load(f)       #記錄對所有feature的partial edge  
    row_idx = []
    col_idx = []
    data = []
    for i,adjacent_v in enumerate(adjacent_s):
        if i ==0 :
            continue    
        row_idx.append(i-1)
        col_idx.append(i-1)
        data.append(float(para_num))
        for v in adjacent_v:
            if v == 0:
                continue
            row_idx.append(i-1)
            col_idx.append(v-1)
            data.append(-1.)
    row_idx_arr = np.array(row_idx)
    col_idx_arr = np.array(col_idx)
    data_arr = np.array(data)
    sparse_matrix = sps.csr_matrix((data, (row_idx, col_idx)), shape=(v_num-1,v_num-1), dtype=np.float32)
    return sparse_matrix

for i in range(1,15):
    folder_path = 'A_matrix'
    A = create_A_matrix(i)
    with open (f'A_matrix/A_matrix_{i}.pkl', 'wb') as f:
        pickle.dump(A, f)