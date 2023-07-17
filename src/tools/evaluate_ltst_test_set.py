import os
import sys

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 将项目根目录路径添加到系统路径
sys.path.append(project_dir)

import torch
import numpy as np
from utils.data_preprocessing import MyDataset
from models.cnn_model import Net
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score
backbone           = "resnet50"
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_set_path_list = ["data_adjust\ltst_part_s306_single.data", "data_adjust\ltst_part_s307_single.data"]
test_dataset       = MyDataset(test_set_path_list)
test_loader        = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
model              = Net(backbone_name=backbone).to(device)
model_path         = os.path.join("trained_models", backbone)
# model_list = os.listdir(model_path)

model_list = os.listdir(model_path)
model_list = ["resnet50_epoch_48.pth"]

for temp_model in model_list:
    label_list     = []
    pred_list      = []
    div_list       = []
    div_label_list = []
    loop           = tqdm(test_loader)


    model.load_state_dict(torch.load(os.path.join(model_path, temp_model)))
    print("model: {}".format(temp_model))
    model.eval()

    for x, y in loop:
        x, y              = x.to("cuda"), y.to("cuda")
        wave              = x.unsqueeze(1)
        label             = y[:,:3]
        mask              = y[:,4].unsqueeze(1)
        y1, y2            = model(wave)
        # y3是y2与y1的差值
        y3                = y2 - y1
        # 将y1, y2, y3拼在一起，得到forward_output
        forward_output    = torch.cat((y1, y2, y3), dim=1)

        IsoJdiv_list = forward_output[:, 1] - forward_output[:, 0]
        IsoJdiv_list = IsoJdiv_list.view(-1).detach().cpu().numpy()

        div_list.extend(IsoJdiv_list)

        div_label = np.array(y[:, 2].tolist())
        div_label_list.extend(div_label)

        # 类别0初始化，0是正常类别， 类别1为st抬高，类别2为st压低，此模块为预测值
        cls=np.zeros((len(IsoJdiv_list),), dtype=int)
        e_rows = np.where((IsoJdiv_list>0.1))
        cls[e_rows]=1
        d_rows = np.where((IsoJdiv_list<-0.1))
        cls[d_rows]=2
        pred_list.extend(cls)
        
        # 此模块为真实值
        cls=np.zeros((len(div_label),), dtype=int)
        e_rows = np.where((div_label>0.1))
        cls[e_rows]=1
        d_rows = np.where((div_label<-0.1))
        cls[d_rows]=2
        label_list.extend(cls)
    
    # 转成numpy
    div_label_list=np.array(div_label_list)
    div_list=np.array(div_list)
    label_list=np.array(label_list)
    pred_list=np.array(pred_list)
    
    # 除去0电位部分数据索引
    good_pos=np.where(div_label_list!=0)
    div_label_list=div_label_list[good_pos]
    div_list=div_list[good_pos]

    label_list=label_list[good_pos]
    pred_list=pred_list[good_pos]

    # 如果阈值相差在0.01mv之内，则两个标签相同
    diff_div=div_label_list-div_list
    index = np.arange(0, len(label_list))
    neq=index[label_list != pred_list]
    thea_diff=np.where(abs(diff_div[neq])<0.01)[0]  # 修正为[0]

    thea_diff=neq[thea_diff]
    pred_list[thea_diff]=label_list[thea_diff]

    print(len(np.where(label_list==0)[0]))
    print(len(np.where(label_list==1)[0]))
    print(len(np.where(label_list==2)[0]))
    print(np.mean(abs(np.array(div_label_list)-np.array(div_list))))

    print(precision_score(label_list, pred_list, average=None))
    print(recall_score(label_list, pred_list, average=None))
