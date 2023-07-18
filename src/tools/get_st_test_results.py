import os
import sys

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 将项目根目录路径添加到系统路径
sys.path.append(project_dir)

from models.cnn_model import Net
from utils.data_preprocessing import MyDataset, custom_collate_fn

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

def evaluate_cls(data, model, save_name, save_path="st_test_result"):
    label_list    = []
    name_list     = []
    div_list      = []
    resample_list = []

    for batch in data:
        y1, y2= model(batch[0].to(device))  # 假设 y1 是 iso 电平，y2 是 st 电平
        # 拼接
        IsoJdiv = torch.cat((y1, y2), dim=1)
        print(IsoJdiv.shape)
        div_list.extend(np.array(IsoJdiv.tolist())[:,1].tolist()) # 将 st 电平添加到 div_list 中
        
        batch1 = torch.flatten(batch[1], end_dim=1)
        print(batch1.shape)
        batch_list = list(batch)
        batch_list[1] = batch1
        
        label_list.extend(y.flatten().tolist())  # 将标签数据添加到 label_list 中

        resample = resample[0]  # 解包 resample 元组，取出其中的张量
        resample = torch.flatten(resample, end_dim=0)
        resample_list.extend(resample.flatten().tolist())  # 将重采样数据添加到 resample_list 中

        name_list.extend(rpos)  # 将 rpos 添加到 name_list 中

    pad = [0] * len(name_list)
    print("生成文件中")

    os.makedirs(save_path, exist_ok=True)
    with open("{}/{}.data".format(save_path, save_name), "wb") as f:
        pickle.dump(list(zip(pad, label_list, div_list, resample_list, name_list, pad)), f)
    print("生成文件完成")


if __name__ == "__main__":

    label_list    = []
    name_list     = []
    div_list      = []
    resample_list = []
    batch_size    = 1024
    backbone      = "resnet50"
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model         = Net(backbone_name=backbone).to(device)
    model_path    = os.path.join("trained_models", backbone)
    model_list    = ["resnet50_epoch_49.pth"]
    st_path       = "st_test_1points" # 预处理好的st数据路径
    for model_name in model_list:
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        model.eval()
        print("模型加载完成")

        # 加载数据
        # test_data_list = [os.path.join(st_path, i) for i in os.listdir(st_path)]
        test_data_list = [os.path.join(st_path, 'European_ST_2_single.data')]
        test_dataset = MyDataset(test_data_list, st=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        print("数据加载完成")

        # 评估模型
        evaluate_cls(test_loader, model, save_name=model_name.split(".")[0])

