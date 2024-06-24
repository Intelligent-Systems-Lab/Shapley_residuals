import matplotlib.pyplot as plt
from feature_name import feature_name
from trans import instance_data_to_subset_list
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
