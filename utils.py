from matplotlib import pyplot as plt
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

# 計算Accuracy, Recall, Precision
def cal_metrics(outputs, targets, num_classes, device):
    
    outputs = torch.softmax(outputs, dim=1)
    targets = torch.argmax(targets, dim=1)
    
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    accuracy_no_avg = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    precision = torchmetrics.Precision(task="multiclass", average='none', num_classes=num_classes).to(device)
    recall = torchmetrics.Recall(task="multiclass", average='none', num_classes=num_classes).to(device)

    accuracy.update(outputs, targets)
    accuracy_no_avg.update(outputs, targets)
    precision.update(outputs, targets)
    recall.update(outputs, targets)

    acc = accuracy.compute()
    acc_no_avg = accuracy_no_avg.compute()
    pre = precision.compute()
    rec = recall.compute()

    acc_0 = acc_no_avg[0]
    acc_1 = acc_no_avg[1]

    rec_0 = rec[0]
    rec_1 = rec[1]

    pre_0 = pre[0]
    pre_1 = pre[1]

    prec_macro = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
    rec_macro = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)
    
    prec = prec_macro(outputs,targets)
    rec = rec_macro(outputs,targets)
    
    return acc, acc_0, acc_1, rec, prec, rec_0, rec_1, pre_0, pre_1


# 繪製訓練與測試模型結果
def draw_pics(record, n, current_save_dir, file_name):
    
    n = n+1
    
    plt.figure(figsize=(15, 10))

    ## Training ##
    # loss
    plt.subplot(4, 2, 1)
    plt.plot(record['train']['loss'])
    plt.axvline(len(record['train']['loss']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # acc
    plt.subplot(4, 2, 3)
    plt.plot(record['train']['acc'], color='black', label='accuracy')
    plt.axvline(len(record['train']['acc']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    
    # acc of classes
    plt.subplot(4, 2, 5)
    plt.plot(record['train']['acc_0'], color='purple', label='A')
    plt.plot(record['train']['acc_1'], color='blue', label='B')
    plt.axvline(len(record['train']['acc_0']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Training acc. of each class')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')

    # rec of classes
    plt.subplot(4, 2, 7)
    plt.plot(record['train']['rec_0'], color='purple', label='A')
    plt.plot(record['train']['rec_1'], color='blue', label='B')
    plt.axvline(len(record['train']['rec_0']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Training recall')
    plt.xlabel('Epoch')
    plt.ylabel('recall')
    plt.legend(loc='upper right')

    ## Testing ##
    # loss
    plt.subplot(4, 2, 2)
    plt.plot(record['test']['loss'])
    plt.axvline(len(record['test']['loss']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # acc
    plt.subplot(4, 2, 4)
    plt.plot(record['test']['acc'], color='black', label='accuracy')
    plt.axvline(len(record['test']['acc']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Testing accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')

    # acc of classes
    plt.subplot(4, 2, 6)
    plt.plot(record['test']['acc_0'], color='purple', label='A')
    plt.plot(record['test']['acc_1'], color='blue', label='B')
    plt.axvline(len(record['test']['acc_0']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Testing acc. of each class')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    
    # rec of classes
    plt.subplot(4, 2, 8)
    plt.plot(record['test']['rec_0'], color='purple', label='A')
    plt.plot(record['test']['rec_1'], color='blue', label='B')
    plt.axvline(len(record['test']['rec_0']) - n, color='red', linestyle='--', label='early stopping')
    plt.title('Testing recall')
    plt.xlabel('Epoch')
    plt.ylabel('recall')
    plt.legend(loc='upper right')
   
    plt.tight_layout()
    plt.savefig(current_save_dir+"/"+file_name+".png")

def initial_record():
    record = {'train':{}, 'test':{}}
    metrics = ['loss', 'acc', 'rec', 'prec', 'rec_0', 'rec_1', 'pre_0', 'pre_1', 'acc_0', 'acc_1']
    for j in record.keys():
        for m in metrics:
            record[j][m] = []
    return record

# 取得EarlyStopping前最後一次模型測試結果
def get_the_last_record(record):
    
    result = initial_record()
    result['train']['loss'].append(record['train']['loss'][-1])
    result['train']['acc'].append(record['train']['acc'][-1])
    result['train']['rec'].append(record['train']['rec'][-1])
    result['train']['prec'].append(record['train']['prec'][-1])
    result['train']['rec_0'].append(record['train']['rec_0'][-1])
    result['train']['rec_1'].append(record['train']['rec_1'][-1])
    result['train']['pre_0'].append(record['train']['pre_0'][-1])
    result['train']['pre_1'].append(record['train']['pre_1'][-1])
    result['train']['acc_0'].append(record['train']['acc_0'][-1])
    result['train']['acc_1'].append(record['train']['acc_1'][-1])
    
    result['test']['loss'].append(record['test']['loss'][-1])
    result['test']['acc'].append(record['test']['acc'][-1])
    result['test']['rec'].append(record['test']['rec'][-1])
    result['test']['prec'].append(record['test']['prec'][-1])
    result['test']['rec_0'].append(record['test']['rec_0'][-1])
    result['test']['rec_1'].append(record['test']['rec_1'][-1])
    result['test']['pre_0'].append(record['test']['pre_0'][-1])
    result['test']['pre_1'].append(record['test']['pre_1'][-1])
    result['test']['acc_0'].append(record['test']['acc_0'][-1])
    result['test']['acc_1'].append(record['test']['acc_1'][-1])
    
    return result

