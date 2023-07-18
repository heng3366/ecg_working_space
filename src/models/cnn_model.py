import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 将项目根目录路径添加到系统路径
sys.path.append(project_dir)
from backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class Net(nn.Module):
    def __init__(self, backbone_name):
        super(Net, self).__init__()
        self.backbone = backbone_name
        # 列举一个字典，用于根据backbone的名字来获取backbone的类
        backbone_dict = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "resnet101": resnet101(),
            "resnet152": resnet152()
        }
        self.backbone = backbone_dict[backbone_name].to("cuda")

    def forward(self, x):
        # x = self.backbone(x)
        # x1, x2, x3 = x
        # x1 = self.relu(self.fc1(x1))
        # x2 = self.sigmoid(self.fc2(x2))
        # x3 = self.sigmoid(self.fc3(x3))
        y1, y2 = self.backbone(x)
        return y1, y2


if __name__ == '__main__':
    model = Net(num_classes=3).to("cuda")
    print(model)
    # 输入ecg数据
    x = torch.randn(512, 1, 256).to("cuda")
    # 前向传播
    y1, y2, y3 = model(x)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)


        
        


        