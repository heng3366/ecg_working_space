# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 将项目根目录路径添加到系统路径
sys.path.append(project_dir)

from utils.data_preprocessing import MyDataset
from utils.seed_everything import set_seed
from models.cnn_model import Net

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

from datetime import datetime
from tqdm import tqdm

# 固定随机数种子
set_seed(42)

def train(args):
    log_path       = args.log_save_path
    os.makedirs(log_path, exist_ok=True)
    # 获取当前时间
    current_time   = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 使用 args 中的参数设置训练过程,获取当前时间
    logging.basicConfig(filename=os.path.join(log_path, current_time+'.log'), level=logging.INFO, encoding="utf-8")

    # 模型选择
    backbone       = args.backbone

    # 数据加载及保存路径
    data_path      =  args.data_path #nas盘数据
    data_path_list = [os.path.join(data_path, name) for name in os.listdir(data_path)][:7]
    model_save_dir = args.model_save_dir

    # 超参数
    batch_size    = args.batch_size
    num_epochs    = args.num_epochs
    gpu_indices   = args.gpu
    devices       = [torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu") for index in gpu_indices]
    train_ratio   = args.train_ratio
    val_ratio     = args.val_ratio
    learning_rate = args.learning_rate
    scaler        = GradScaler()

    # 数据加载
    dataset     = MyDataset(data_path_list)
    train, val  = random_split(dataset, [int(len(dataset)*train_ratio), len(dataset)-int(len(dataset)*train_ratio)])
    train_loader= DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader  = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 日志存储路径
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # 打印所有的args参数到logging日志文件
    logging.info("All args parameter:")
    for arg in vars(args):
        logging.info(f"   {arg}: {getattr(args, arg)}")

    # 将超参数写入到logging日志文件
    logging.info("Hyper Parameters:")
    logging.info(f"   batch_size: {batch_size}")
    logging.info(f"   learning_rate: {learning_rate}")
    logging.info(f"   num_epochs: {num_epochs}")
    logging.info(f"   device: {devices}")
    logging.info(f"   train_ratio: {train_ratio}")
    logging.info(f"   val_ratio: {val_ratio}")
    logging.info(f"   backbone: {backbone}")
    
    # 网络结构写入logging日志文件
    logging.info("Model Structure:")
    logging.info("    " + str(Net(backbone_name=backbone).to(devices[0])))

    # 每轮迭代数据次数写入logging日志文件
    logging.info("DataLoader:")
    logging.info(f"   train_loader: {len(train_loader)}")
    logging.info(f"   val_loader: {len(val_loader)}")

    # 模型开始训练写入logging日志文件
    logging.info("Start Training...")
    logging.info("=======================================")


    # 加载模型及优化器
    model = Net(backbone_name=backbone).to(devices[0])
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # 按照设置的epoch数进行循环
    for epoch in range(num_epochs):
        # 每轮训练开始时间
        epoch_start_time = datetime.now()
        # 记录当前epoch到logging日志文件
        logging.info(f"Epoch: {epoch+1}/{num_epochs} | Start Time: {epoch_start_time.strftime('%Y-%m-%d-%H-%M-%S')}")
        # 记录当前学习率到logging日志文件
        logging.info(f"     || Current Learning Rate: {scheduler.get_last_lr()[0]}")

        div_label_list   = []
        label_list       = []
        pred_list        = []
        div_list         = []
        total_epoch_loss = 0

        model.train()
        total_epoch_loss = 0

        for x, y in train_loader:
            # x, y   = x.to(devices), y.to(devices)
            x, y            = x.to(devices[0]), y.to(devices[0])
            wave   = x.unsqueeze(1)
            label  = y[:,:3]
            label1 = label[:, 0].unsqueeze(1)
            label2 = label[:, 1].unsqueeze(1)

            with autocast():
                pred1, pred2      = model(wave)
                loss1             = F.mse_loss(pred1, label1)
                loss2             = F.mse_loss(pred2, label2)
                loss              = loss1 + loss2
                total_epoch_loss += loss.item()

            # preision, recall, f1-score评测
            forward_output = torch.cat([pred1, pred2], dim=1)
            IsoJdiv_list   = forward_output[:, 1] - forward_output[:, 0]
            IsoJdiv_list   = IsoJdiv_list.view(-1).detach().cpu().numpy()
            div_label      = np.array(y[:, 2].tolist())
            div_list.extend(IsoJdiv_list)
            div_label_list.extend(div_label)

            # 类别0初始化，0是正常类别， 类别1为st抬高，类别2为st压低，此模块为预测值
            cls         = np.zeros((len(IsoJdiv_list),), dtype=int)
            e_rows      = np.where((IsoJdiv_list>0.1))
            cls[e_rows] = 1
            d_rows      = np.where((IsoJdiv_list<-0.1))
            cls[d_rows] = 2
            pred_list.extend(cls)

            # 此模块为真实值
            cls         = np.zeros((len(div_label),), dtype=int)
            e_rows      = np.where((div_label>0.1))
            cls[e_rows] = 1
            d_rows      = np.where((div_label<-0.1))
            cls[d_rows] = 2
            label_list.extend(cls)

            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 跟新学习率
        scheduler.step()

        # 记录每个epoch的平均loss到logging日志文件
        logging.info(f"     || Epoch: {epoch+1}/{num_epochs} | Train Loss: {total_epoch_loss/len(train_loader):.6f}")

        # 转成numpy
        div_label_list = np.array(div_label_list)
        div_list       = np.array(div_list)
        label_list     = np.array(label_list)
        pred_list      = np.array(pred_list)

        # 去除0电位部分数据索引
        good_pos       = np.where(div_label_list!=0)
        div_label_list = div_label_list[good_pos]
        div_list       = div_list[good_pos]
        label_list     = label_list[good_pos]
        pred_list      = pred_list[good_pos]

        # 如果阈值相差在0.01mv之内，则两个标签相同
        diff_div             = div_label_list-div_list
        index                = np.arange(0, len(label_list))
        neq                  = index[label_list != pred_list]
        thea_diff            = np.where(abs(diff_div[neq])<0.01)[0]  # 修正为[0]
        thea_diff            = neq[thea_diff]
        pred_list[thea_diff] = label_list[thea_diff]

        # 将precision, recall, f1-score写入logging日志文件，弄成表格形式
        logging.info(f"     || Train Precision: {precision_score(label_list, pred_list, average=None)}")
        logging.info(f"     || Train Recall: {recall_score(label_list, pred_list, average=None)}")
        logging.info(f"     || Train F1-Score: {f1_score(label_list, pred_list, average=None)}")

        # 验证
        model.eval()
        total_val_loss = 0
        label_list     = []
        pred_list      = []
        div_list       = []
        div_label_list = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y            = x.to(devices[0]), y.to(devices[0])
                wave            = x.unsqueeze(1)
                label           = y[:,:3]
                label1          = label[:, 0].unsqueeze(1)
                label2          = label[:, 1].unsqueeze(1)

                pred1, pred2     = model(wave)
                loss1            = F.mse_loss(pred1, label1)
                loss2            = F.mse_loss(pred2, label2)
                loss             = loss1 + loss2
                total_val_loss  += loss.item()

                # 计算precision, recall, f1-score
                forward_output = torch.cat([pred1, pred2], dim=1)
                IsoJdiv_list   = forward_output[:, 1] - forward_output[:, 0]
                IsoJdiv_list   = IsoJdiv_list.view(-1).detach().cpu().numpy()
                div_label      = np.array(y[:, 2].tolist())

                div_list.extend(IsoJdiv_list)
                div_label_list.extend(div_label)

                # 类别0初始化，0是正常类别， 类别1为st抬高，类别2为st压低，此模块为预测值
                cls         = np.zeros((len(IsoJdiv_list),), dtype=int)
                e_rows      = np.where((IsoJdiv_list>0.1))
                cls[e_rows] = 1
                d_rows      = np.where((IsoJdiv_list<-0.1))
                cls[d_rows] = 2
                pred_list.extend(cls)

                # 此模块为真实值
                cls         = np.zeros((len(div_label),), dtype=int)
                e_rows      = np.where((div_label>0.1))
                cls[e_rows] = 1
                d_rows      = np.where((div_label<-0.1))
                cls[d_rows] = 2
                label_list.extend(cls)

            # loss写入logging日志文件
            logging.info(f"     || Epoch: {epoch+1}/{num_epochs} | Val Loss: {total_val_loss/len(val_loader):.6f}")
            
            # 转成numpy
            div_label_list = np.array(div_label_list)
            div_list       = np.array(div_list)
            label_list     = np.array(label_list)
            pred_list      = np.array(pred_list)

            # 去除0电位部分数据索引
            good_pos       = np.where(div_label_list!=0)
            div_label_list = div_label_list[good_pos]
            div_list       = div_list[good_pos]
            label_list     = label_list[good_pos]
            pred_list      = pred_list[good_pos]

            # 如果阈值相差在0.01mv之内，则两个标签相同
            diff_div             = div_label_list-div_list
            index                = np.arange(0, len(label_list))
            neq                  = index[label_list != pred_list]
            thea_diff            = np.where(abs(diff_div[neq])<0.01)[0]  # 修正为[0]
            thea_diff            = neq[thea_diff]
            pred_list[thea_diff] = label_list[thea_diff]

            # 将precision, recall, f1-score写入logging日志文件，弄成表格形式
            logging.info(f"     || Val Precision: {precision_score(label_list, pred_list, average=None)}")
            logging.info(f"     || Val Recall: {recall_score(label_list, pred_list, average=None)}")
            logging.info(f"     || Val F1-Score: {f1_score(label_list, pred_list, average=None)}")

            # 保存模型
            model_name = f"{backbone}_{epoch+1}.pth"
            save_name  = os.path.join(model_save_dir, model_name)
            torch.save(model.state_dict(), save_name)

            # 打印存入的模型名称
            logging.info(f"     || Save Model: {model_name}")
            # 日志打印多余3行空行
            logging.info("\n\n\n")



 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--data_path", type=str, default="//192.168.2.8/xjk/Algorithm/ECG_ST/transformer-data/训练数据/ltst_single_data", help="data path")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val ratio")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="num epochs")
    parser.add_argument("--backbone", type=str, default="resnet18", help="backbone")
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU device indices to use (default: [0])")
    parser.add_argument("--model_save_dir", type=str, default="model_save", help="model save dir")
    parser.add_argument("--log_save_dir", type=str, default="log_save", help="log save dir")
    parser.add_argument("--log_save_path", type=str, default="train_log", help="log name")
    args = parser.parse_args()

    try:
        # 执行可能会抛出异常的代码
        train(args)
    except Exception as e:
        # 捕获异常并写入日志文件
        logging.exception("An exception occurred during training:")