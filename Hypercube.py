import numpy as np
import scipy.sparse as sps
import itertools
import bisect
import math
import json
import os
import pickle
from trans import generate_all_subsets

class Hypercube_:                #超立方體
    '''
    A class to create a hypercube object which stores values on vertices
    and values on the edges between neighboring vertices
    '''    
    #輸入維度、(點鍵值)、(點值)
    def __init__(self, n_vertices, vertex_keys = None, vertex_values = None):   
        self.n_vertices = n_vertices
        self.v_num = 2**self.n_vertices     #點數
        self.V_subsets =  generate_all_subsets(self.n_vertices)  #所有子集包含空集，即所有點
        self.V = [i for i in range(self.v_num)]     #點的index
        self.V_value = []
        self.E = []         #記錄邊(點的index)
        self.E_value = []   #記錄邊值(有向邊)
        self.b_i = []
        self.v_i = []
    def set_vertex_values(self, exp_file):         #設置點值
        with open(exp_file,'r') as f:
            for line in f:
                self.V_value.append(float(line.strip()))       
        #檔案可以改成pkl檔，更好處理
        self._calculate_edges()         
        
    def _calculate_edges(self):                 #計算邊值
        
        # calculate the usual gradients: the difference between neighboring edges
        with open(f'vertex_adjacent/vertex_adjacent_{self.n_vertices}.pkl', 'rb') as f:
            self.adjacent_list = pickle.load(f)
        with open(f'/hcds_vol/private/luffy/GANGAN-master/Shap_residual/partial_edges/partial_edges_{self.n_vertices}.pkl', 'rb') as f:
            self.partial_edges = pickle.load(f)       #記錄對所有feature的partial edge  
        for i in range(self.v_num):
            for adjacent in self.adjacent_list[i]:
                self.E.append([i,adjacent])            #記錄所有邊
                self.E_value.append(self.V_value[adjacent] - self.V_value[i])      #邊值會記錄成有向邊
    #create matrix for Hypercube
    
    def trans_to_matrix(self):                #超立方體轉換成矩陣
        with open(f'A_matrix/A_matrix_{self.n_vertices}.pkl', 'rb') as f:
            self.A = pickle.load(f)
        for i, partial_edge in enumerate(self.partial_edges):           #從14個特徵中挑選
            b = np.array([0.]*self.v_num) 
            for edge in partial_edge:      #
                b[edge[0]] += self.V_value[edge[0]] - self.V_value[edge[1]]
                b[edge[1]] += self.V_value[edge[1]] - self.V_value[edge[0]]
            self.b_i.append(b[1:])
        return self.b_i
    def calculate_shap_residual(self):
        for b  in self.b_i:
            #self.v_i.append(sps.linalg.spsolve(self.A,b))
            vi = np.insert(sps.linalg.spsolve(self.A,b),0,0.)
            self.v_i.append(vi)
            print('finish_one')
        return self.v_i

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #minimize the function (gradient_x - partial_gradient_i)^2 最小化l2_norm
    def shapley_residuals_in_matrix(self):
            derivative_i  = np.full((self.v_num,self.v_num),0.)      #計算微分後得到的方程式矩陣 A
            b_i = np.array([0.]*self.v_num)                          #得到Ax = b 的 b向量值
            for j  in range(self.v_num):                            #對矩陣的每個點
                for k in range(self.v_num):
                    if np.isnan(self.partial_gradient_matrix[j][k]):    #不用計算nan
                        continue
                    #elif j == 0 or k ==0:                               #如果是跟原點相鄰
                     #   derivative_i[j][j] += 1.                       #係數+1
                      #  b_i[j] += - self.partial_gradient_matrix[j][k]     #x_j-x_i-partial_gradient
                    else:                                               
                        derivative_i[j][j] += 1.                         
                        derivative_i[j][k] += -1.
                        b_i[j] += - self.partial_gradient_matrix[j][k]      # j -k
            A = derivative_i[1:,1:]                                    #只要x_i for i!=0
            
            B = b_i[1:]                                                #保留b_i
            res = 0.
            A_inverse = np.linalg.inv(A)                               #算inverse matrix
            vi = np.insert(np.dot(A_inverse,B),0,0.)                    #vi  = b/A #開頭補0
            vi_V =  [np.array([])] + all_subsets(self.n_vertices)
            vi_V_value = {str(v) : 0. for v in vi_V} 
            res_dic = {}
            for k,v in enumerate(vi_V):               
                vi_V_value[str(v)] = vi[k]
            self.vi_vector = np.array([])                              #比較向量#2
            for i, v in enumerate(vi_V):
                for j,_v in enumerate(vi_V[i+1:]):
                    if self._vertices_form_a_valid_edge(v, _v):
                        self.vi_matrix[i][i+j+1] = vi_V_value[str(_v)] - vi_V_value[str(v)]
                        self.vi_matrix[i+j+1][i] = vi_V_value[str(v)] - vi_V_value[str(_v)]
                        self.res_matrix[i][i+j+1] = self.partial_gradient_matrix[i][i+j+1] - self.vi_matrix[i][i+j+1]
                        self.res_matrix[i+j+1][i] = self.partial_gradient_matrix[i+j+1][i] - self.vi_matrix[i+j+1][i]
                        res += (self.res_matrix[i][i+j+1])**2
                        res_dic[str(v)+'->'+str(_v)] = self.res_matrix[i][i+j+1]
                        if (i,i+j+1) in self.my_list:
                            self.vi_vector = np.append(self.vi_vector,self.vi_matrix[i][i+j+1])     #比較向量#2
            #print(self.my_list)
            #print(self.partial_gradient_vector,self.vi_vector)
            vector_A = self.weight_vector*self.partial_gradient_vector
            vector_B = self.weight_vector*self.vi_vector
            cos_sim = self.cos_sim(vector_A,vector_B)
            print('cos_sim = ',cos_sim)
            self.sv += vi[-1]
            print('shapley_value: ',vi[-1],'residual sum: ',res)
            #print(self.sv)
            #print('residuals_sum:',res,'shapley_value: ',vi)
            res =  (res/self.partial_gradient_norm)**0.5
            print('norm: ',res)
            
            return vi_V_value, res_dic, self.partial_gradient_norm
    

            

