import numpy as np
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
def padding_0(df, para_num, param_group=[2,2,8,2] ): 
    # 將一維參數matrix擴展為4維
    data_arr = df.to_numpy()
    result = []
    for i in range(len(data_arr)):
        new_arr = np.zeros((4,para_num))
        start_idx = 0
        end_idx = 0
        for j in range(len(new_arr)):
            end_idx = start_idx + param_group[j] 
            new_arr[j][start_idx:end_idx] = data_arr[i][start_idx:end_idx]
            start_idx = end_idx
        result.append(new_arr)
    result = pd.DataFrame({'X': [result[i] for i in range(len(result))]})
    return result

def send_to_model(data_standardized_df,para_num,model,device):
    data_4d = padding_0(data_standardized_df,para_num=para_num)
    #data_4d.to_csv('checkout_df1.csv')
    data_4d_array = np.array([e for entry in data_4d.values for e in entry])
    data_4d_tensor = torch.tensor(data_4d_array,dtype=torch.float)
    batch_data = data_4d_tensor.to(device)
    #my_dataset = TensorDataset(data_4d_tensor)
    #batch_size = min(256, int(data_4d_tensor.size()[0]))
    #batch_size = 256
    #my_loader = DataLoader(my_dataset, batch_size=batch_size,num_workers=4)
    data_output = []
    with torch.no_grad():
        #for batch_data in my_loader:
        #移到設備
            #batch_data = batch_data[0].to(device)
        # 進行推理
        batch_output = model(batch_data)
        probs = (torch.nn.functional.softmax(batch_output, dim=1))
        # 保存輸出
        data_output += probs
    data_output_arr = np.array([output.cpu().numpy()[0] for output in data_output])
    data_expectation_out = data_output_arr.mean()
    return data_expectation_out

def save_expectation_as_txt(data,idx):
    folder_path = '/hcds_vol/private/luffy/GANGAN-master/Shap_residual/instance_expectation'
    file_path = os.path.join(folder_path, f'expectation_{idx}.txt')
    with open(file_path, 'w') as f:
        for item in data:
            f.write(f"{item}\n")  # 每個元素占一行
            
def save_cube_data(dict,index):
    folder_name = f'cube_result/instance_{index}'
    os.makedirs(folder_name, exist_ok=True)
    
    
    
    