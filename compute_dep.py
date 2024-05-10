import json
def compare_dependency(partial_norm_list,feature_list,pair):             

    '''input json file res_dic_list and partial gradient norm list and subset indexs
       return scled norm for features in subset
    '''
    res_dic_list = []
    subset_partial_norm = 0
    for idx in pair:
        # trans json file into list of dictionary 
        with open('dic_'+feature_list[idx]+'_res.json') as f:
            data = json.load(f)
            res_dic_list.append(data)
        
        subset_partial_norm += partial_norm_list[idx]       
        
    result = res_dic_list[0]
    for idx in range(1,len(pair)):
        for key in result:
            result[key] += res_dic_list[idx][key]
            
    subset_res_norm = 0
    for key in result:
        subset_res_norm += result[key]**2
    
    return (subset_res_norm/subset_partial_norm)**0.5
p_n_list = []
params_list = []
pairs = [0,1,2]
print(compare_dependency(p_n_list,params_list,pairs))