import numpy as np 
import warnings
import itertools
import pandas as pd
import torch
from tqdm import tqdm
from trans import generate_all_subsets
from sklearn.preprocessing import StandardScaler
from auo_utils import send_to_model, save_expectation_as_txt
# 過濾掉FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


#超參數、路徑設定
s1_model_path = '/hcds_vol/private/luffy/GANGAN-master/model/predictor/stage_1_checkpoint.pth'
alldata_path = '/hcds_vol/private/luffy/GANGAN-master/Shap_residual/data_with_output.csv'

class DataProcessor():
    #處理數據
    def __init__(self,para_num,model_path,device):
        self.para_num = para_num            #參數數量
        self.Coalitions = generate_all_subsets(self.para_num)      #創建所有子集
        self.scaler = StandardScaler()      #可以先固定scaler，不需要每次都傳入data重新做scaler ==>all_data的scaler
        self.Output_mean = 0                #sample_df的Output平均 ==>  sample_data的平均
        self.impact_list = [0]
        self.model = torch.load(model_path).to(device)
        self.model.eval()
        
    def build_background(self,selected_data_df):
        self.background_data_df = selected_data_df.sample(n=1000,random_state=42) #建立背景資料集
        self.Output_mean = self.background_data_df['Output'].mean() #算背景資料集output平均    
        return self.background_data_df  
    
    def fit_transform(self,all_data_df):                      
        data_without_output_df = all_data_df.drop(columns=['Output'])   #去掉output欄做fit
        standardized_data_df = pd.DataFrame(self.scaler.fit_transform(data_without_output_df))     #fit scaler參數、獲得standardized後的資料集
        return standardized_data_df    #回傳standardized後的資料集
    
    def calculate_expectation(self,instance_data,model_path,device):
        
        for coalition in tqdm(self.Coalitions):
            if len(coalition)!=0:
                synth = self.background_data_df.copy()    #用copy()才不會更改原本的dataframe
                synth.iloc[:,coalition] = instance_data.iloc[coalition] #
                if 'Output' in synth.columns:
                    synth = synth.drop(columns=['Output'])
                synth_standardized_df = pd.DataFrame(self.scaler.transform(synth))
                Exp = send_to_model(synth_standardized_df,para_num=self.para_num,model=self.model,device=device)
                impact = Exp - self.Output_mean
                self.impact_list.append(impact)
        return self.impact_list
    
alldata_df = pd.read_csv(alldata_path)
A = DataProcessor(14,s1_model_path,device='cuda:1')
new_data = A.fit_transform(alldata_df)
selected_data = alldata_df.copy()
selected_data = selected_data[(selected_data['PUMP_low']<20000) & 
                              (selected_data['PUMP_high']>20000) & 
                              (selected_data['VENT_low']<10000) & 
                              (selected_data['VENT_high']>10000) &
                              (selected_data['NH3_TREAT_-RF_FREQ-max']>13800)&
                              (selected_data['NH3_TREAT_-RF_FREQ-mean']<13800)]
background_df = A.build_background(selected_data)
print('OK')
for index in tqdm(range(100,200)):
    instance = alldata_df.iloc[index]
    pro_list = A.calculate_expectation(instance,s1_model_path,'cuda:1')
    save_expectation_as_txt(pro_list,index)
