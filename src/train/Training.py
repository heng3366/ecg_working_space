import os
import sys

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

if __name__ == "__main__":
    data_path      = "data_adjust"
    data_path_list = [os.path.join(data_path, name) for name in os.listdir(data_path)][:7]
    dataset        = MyDataset(data_path_list)
    train_ratio    = 0.9  # 训练集占总数据集的比例
    val_ratio      = 0.1    # 验证集占总数据集的比例
    batch_size     = 1024 
    learning_rate  = 1e-4
    num_epochs     = 300
    backbone       = "resnet50"
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model          = Net(backbone_name=backbone).to(device)
    scaler         = GradScaler() # 混合精度训练
    optimizer      = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    train, val     = random_split(dataset, [int(len(dataset)*train_ratio), len(dataset)-int(len(dataset)*train_ratio)])
    train_loader   = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader     = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    file_path      = os.path.join("logs/train_log", backbone)
    save_dir       = os.path.join("trained_models", backbone)
    now            = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"    # 当前时间，用于记录日志

    os.makedirs(file_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    #按照设置的epoch数进行循环
    with open(os.path.join(file_path, now), "w", encoding="utf-8") as file:

        # 模型结构和超参数写入到日志文件
        file.write("Model Structure:\n")
        file.write("    "+str(Net(backbone_name=backbone).to("cuda")))
        file.write("\n")
        file.write("Hyper Parameters:\n")
        file.write(f"   batch_size: {batch_size}\n")
        file.write(f"   learning_rate: {learning_rate}\n")
        file.write(f"   num_epochs: {num_epochs}\n")
        file.write(f"   device: {device}\n")
        file.write(f"   optimizer: {optimizer}\n")
        file.write(f"   scheduler: {scheduler}\n")
        file.write("\n")
        file.write("Training Process:\n")
        file.flush()  # 刷新缓冲区，确保内容被写入文件

        for epoch in range(num_epochs):
            div_label_list   = []
            label_list       = []
            pred_list        = []
            div_list         = []
            total_epoch_loss = 0
            total_test_loss  = 0


            model.train()
            total_epoch_loss = 0

            bar = tqdm(desc=f" | Epoch: {epoch+1}", postfix=dict, mininterval=0.3)

            for x, y in train_loader:
                x, y   = x.to(device), y.to(device)
                wave   = x.unsqueeze(1)
                label  = y[:,:3]
                label1 = label[:, 0].unsqueeze(1)
                label2 = label[:, 1].unsqueeze(1)

                with autocast(): 
                    y1, y2            = model(wave)
                    loss1             = F.mse_loss(y1, label1)
                    loss2             = F.mse_loss(y2, label2)
                    loss              = loss1 + loss2
                    total_epoch_loss += loss.item()

                # preision, recall, f1-score评测
                forward_output = torch.cat([y1, y2], dim=1)
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

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                bar.set_description(f"epoch: {epoch} | Train Loss: {loss.item():.6f}")
                bar.update()

            bar.close()

            # 更新学习率
            scheduler.step()

            # 打印每个epoch的平均训练损失
            mean_train_loss = total_epoch_loss / len(train_loader)
            print(f"Epoch: {epoch+1} | Mean Train Loss: {mean_train_loss:.6f}")
            
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

            print(precision_score(label_list, pred_list, average=None))
            print(recall_score(label_list, pred_list, average=None))
            print(f1_score(label_list, pred_list, average=None))
            # 将类别0、1、2的数量写入文件，以及precision和recall和f1-score
            if epoch == 0:
                file.write(f"训练集情况：\n")
                file.write(f"类别0数量: {len(np.where(label_list==0)[0])} | 类别1数量: {len(np.where(label_list==1)[0])} | 类别2数量: {len(np.where(label_list==2)[0])}\n")
            
            file.write(f"类别0的precision: {precision_score(label_list, pred_list, average=None)[0]} | 类别1的precision: {precision_score(label_list, pred_list, average=None)[1]} | 类别2的precision: {precision_score(label_list, pred_list, average=None)[2]}\n")
            file.write(f"类别0的recall: {recall_score(label_list, pred_list, average=None)[0]} | 类别1的recall: {recall_score(label_list, pred_list, average=None)[1]} | 类别2的recall: {recall_score(label_list, pred_list, average=None)[2]}\n")
            file.write(f"类别0的f1-score: {f1_score(label_list, pred_list, average=None)[0]} | 类别1的f1-score: {f1_score(label_list, pred_list, average=None)[1]} | 类别2的f1-score: {f1_score(label_list, pred_list, average=None)[2]}\n")
            # 缓冲区刷新
            file.flush()

            # 验证集
            model.eval()

            total_val_loss = 0
            label_list     = []
            pred_list      = []
            div_list       = []
            div_label_list = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y                = x.to(device), y.to(device)
                    wave                = x.unsqueeze(1)
                    label               = y[:,:3]
                    label1              = label[:, 0].unsqueeze(1)
                    label2              = label[:, 1].unsqueeze(1)

                    y1, y2              = model(wave)
                    loss1               = F.mse_loss(y1, label1)
                    loss2               = F.mse_loss(y2, label2)
                    loss                = loss1 + loss2
                    total_val_loss    += loss.item()

                    # 计算prison和recall
                    forward_output = torch.cat([y1, y2], dim=1)
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
                    
            # loss写入文件
            file.write(f"val loss: {total_test_loss/len(val_loader)}\n")
            file.flush()

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

            if epoch == 0:
                print(len(np.where(label_list==0)[0]))
                print(len(np.where(label_list==1)[0]))
                print(len(np.where(label_list==2)[0]))
                print(np.mean(abs(np.array(div_label_list)-np.array(div_list))))

            print(precision_score(label_list, pred_list, average=None))
            print(recall_score(label_list, pred_list, average=None))
            print(f1_score(label_list, pred_list, average=None))
            
            # 将类别0、1、2的数量写入文件，以及precision和recall和f1-score
            if epoch == 0:
                file.write(f"测试集情况：\n")
                file.write(f"类别0数量: {len(np.where(label_list==0)[0])} | 类别1数量: {len(np.where(label_list==1)[0])} | 类别2数量: {len(np.where(label_list==2)[0])}\n")
            file.write(f"类别0的precision: {precision_score(label_list, pred_list, average=None)[0]} | 类别1的precision: {precision_score(label_list, pred_list, average=None)[1]} | 类别2的precision: {precision_score(label_list, pred_list, average=None)[2]}\n")
            file.write(f"类别0的recall: {recall_score(label_list, pred_list, average=None)[0]} | 类别1的recall: {recall_score(label_list, pred_list, average=None)[1]} | 类别2的recall: {recall_score(label_list, pred_list, average=None)[2]}\n")
            file.write(f"类别0的f1-score: {f1_score(label_list, pred_list, average=None)[0]} | 类别1的f1-score: {f1_score(label_list, pred_list, average=None)[1]} | 类别2的f1-score: {f1_score(label_list, pred_list, average=None)[2]}\n")
            # 缓冲区刷新
            file.flush()
            
            # 打印每个epoch的验证损失
            mean_val_loss = total_val_loss / len(val_loader)
            print(f" | Epoch: {epoch+1} | Mean Validation Loss: {mean_val_loss:.6f}")                  

            # 将损失值追加到文件中
            file.write(f"Epoch: {epoch+1} | 训练损失: {mean_train_loss:.6f} | 验证损失: {mean_val_loss:.6f}\n")
            file.flush()  # 刷新缓冲区，确保内容被写入文件

            # 保存模型
            save_name = os.path.join(save_dir, "{}_epoch_{}.pth".format(backbone, epoch+1))
            torch.save(model.state_dict(), save_name)
