import os
import sys
from datetime import datetime

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_dir = os.path.dirname(current_dir)
# 将项目根目录路径添加到系统路径
sys.path.append(project_dir)

from utils.data_preprocessing import MyDataset
from models.cnn_model import Net
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
# 固定随机数种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

if __name__ == "__main__":
    data_path      = "data_adjust"
    data_path_list = [os.path.join(data_path, name) for name in os.listdir(data_path)][:-2]
    dataset        = MyDataset(data_path_list)
    batch_size     = 1024
    learning_rate  = 1e-4
    num_epochs     = 50
    n_folds        = 5
    kfold          = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    fold           = 0
    patience       = 15
    no_improvement = 0
    backbone       = "resnet18"
    file_path      = os.path.join("logs/train_log", backbone)
    save_dir       = os.path.join("trained_models", backbone)
    os.makedirs(save_dir, exist_ok=True)

    # 创建列表来存储每个折的损失值
    train_loss_records = [[] for _ in range(n_folds)]
    val_loss_records   = [[] for _ in range(n_folds)]

    # 当前时间，用于记录日志
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
    os.makedirs(file_path, exist_ok=True)
    
    # 用于记录每个epoch的最佳权重
    best_weights_per_epoch = []

    with open(os.path.join(file_path, now), "w", encoding="utf-8") as file:
        
        # 模型结构和超参数写入到日志文件
        file.write(str(Net(backbone_name=backbone).to("cuda")))
        file.write("\n")
        file.write(f"batch_size: {batch_size}\n")
        file.write(f"learning_rate: {learning_rate}\n")
        file.write(f"num_epochs: {num_epochs}\n")
        file.write(f"n_folds: {n_folds}\n")
        file.write(f"patience: {patience}\n")
        file.write(f"no_improvement: {no_improvement}\n")
        file.write("\n")
        file.flush()  # 刷新缓冲区，确保内容被写入文件


        for train_idx, val_idx in kfold.split(dataset):
            best_val_loss  = float("inf") #每一折都要从头开始记early stopping
            fold          += 1

            print(f"Fold: {fold}")
            file.write(f"折数: {fold}\n")
            file.flush()  # 刷新缓冲区，确保内容被写入文件

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset   = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

            model     = Net(backbone_name=backbone).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            scaler    = GradScaler()

            for epoch in range(num_epochs):
                div_label_list = []
                label_list=[]
                pred_list=[]
                div_list=[]
                # 训练
                model.train()
                total_epoch_loss = 0
                bar = tqdm(desc=f"Fold: {fold} | Epoch: {epoch+1}", postfix=dict, mininterval=0.3)
                for x, y in train_loader:
                    x, y = x.to("cuda"), y.to("cuda")
                    wave = x.unsqueeze(1)
                    label = y[:, :3]
                    mask = y[:, 4].unsqueeze(1)

                    with autocast():
                        y1, y2 = model(wave)
                        label1 = label[:, 0].unsqueeze(1)
                        label2 = label[:, 1].unsqueeze(1)

                        loss1 = F.mse_loss(y1, label1)
                        loss2 = F.mse_loss(y2, label2)
                        loss  = loss1 + loss2

                        total_epoch_loss += loss.item()

                    forward_output = torch.cat([y1, y2], dim=1)
                    IsoJdiv_list = forward_output[:, 1] - forward_output[:, 0]
                    IsoJdiv_list = IsoJdiv_list.view(-1).detach().cpu().numpy()

                    div_list.extend(IsoJdiv_list)

                    div_label = np.array(y[:, 2].tolist())
                    div_label_list.extend(div_label)

                    #类别0初始化，0是正常类别， 类别1为st抬高，类别2为st压低，此模块为预测值
                    cls=np.zeros((len(IsoJdiv_list),), dtype=int)
                    e_rows = np.where((IsoJdiv_list>0.1))
                    cls[e_rows]=1
                    d_rows = np.where((IsoJdiv_list<-0.1))
                    cls[d_rows]=2
                    pred_list.extend(cls)

                    #此模块为真实值
                    cls=np.zeros((len(div_label),), dtype=int)
                    e_rows = np.where((div_label>0.1))
                    cls[e_rows]=1
                    d_rows = np.where((div_label<-0.1))
                    cls[d_rows]=2
                    label_list.extend(cls)  

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    bar.set_description(f"Fold: {fold} | Train Loss: {loss.item():.6f}")
                    bar.update()

                bar.close()

                # 打印每个epoch的平均训练损失
                mean_train_loss = total_epoch_loss / len(train_loader)
                print(f"Fold: {fold} | Epoch: {epoch+1} | Mean Train Loss: {mean_train_loss:.6f}")

                #转成numpy
                div_label_list=np.array(div_label_list)
                div_list=np.array(div_list)
                label_list=np.array(label_list)
                pred_list=np.array(pred_list)

                #除去0电位部分数据索引
                good_pos=np.where(div_label_list!=0)
                div_label_list=div_label_list[good_pos]
                div_list=div_list[good_pos]

                label_list=label_list[good_pos]
                pred_list=pred_list[good_pos]

                #如果阈值相差在0.01mv之内，则两个标签相同
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
                #打印f1-score
                print(f1_score(label_list, pred_list, average=None))
                # 将类别0、1、2的数量写入文件，以及precision和recall和f1-score
                file.write(f"训练集情况：\n")
                file.write(f"类别0数量: {len(np.where(label_list==0)[0])} | 类别1数量: {len(np.where(label_list==1)[0])} | 类别2数量: {len(np.where(label_list==2)[0])}\n")
                file.write(f"类别0的precision: {precision_score(label_list, pred_list, average=None)[0]} | 类别1的precision: {precision_score(label_list, pred_list, average=None)[1]} | 类别2的precision: {precision_score(label_list, pred_list, average=None)[2]}\n")
                file.write(f"类别0的recall: {recall_score(label_list, pred_list, average=None)[0]} | 类别1的recall: {recall_score(label_list, pred_list, average=None)[1]} | 类别2的recall: {recall_score(label_list, pred_list, average=None)[2]}\n")
                file.write(f"类别0的f1-score: {f1_score(label_list, pred_list, average=None)[0]} | 类别1的f1-score: {f1_score(label_list, pred_list, average=None)[1]} | 类别2的f1-score: {f1_score(label_list, pred_list, average=None)[2]}\n")
                # 缓冲区刷新
                file.flush()

                # 更新学习率
                scheduler.step()
                                
                # 验证
                model.eval()
                total_val_loss = 0
                label_list=[]
                pred_list=[]
                div_list=[]
                div_label_list=[]

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to("cuda"), y.to("cuda")
                        wave = x.unsqueeze(1)
                        label = y[:, :3]
                        mask = y[:, 4].unsqueeze(1)

                        y1, y2 = model(wave)
                        label1 = label[:, 0].unsqueeze(1)
                        label2 = label[:, 1].unsqueeze(1)

                        loss1 = torch.sum(F.mse_loss(y1, label1) * mask) / torch.sum(mask)
                        loss2 = torch.sum(F.mse_loss(y2, label2) * mask) / torch.sum(mask)
                        loss = loss1 + loss2

                        total_val_loss += loss.item()

                        # 计算prison和recall
                        forward_output = torch.cat([y1, y2], dim=1)
                        IsoJdiv_list = forward_output[:, 1] - forward_output[:, 0]
                        IsoJdiv_list = IsoJdiv_list.view(-1).detach().cpu().numpy()

                        div_list.extend(IsoJdiv_list)

                        div_label = np.array(y[:, 2].tolist())
                        div_label_list.extend(div_label)

                        #类别0初始化，0是正常类别， 类别1为st抬高，类别2为st压低，此模块为预测值
                        cls=np.zeros((len(IsoJdiv_list),), dtype=int)
                        e_rows = np.where((IsoJdiv_list>0.1))
                        cls[e_rows]=1
                        d_rows = np.where((IsoJdiv_list<-0.1))
                        cls[d_rows]=2
                        pred_list.extend(cls)
                        
                        #此模块为真实值
                        cls=np.zeros((len(div_label),), dtype=int)
                        e_rows = np.where((div_label>0.1))
                        cls[e_rows]=1
                        d_rows = np.where((div_label<-0.1))
                        cls[d_rows]=2
                        label_list.extend(cls)
                
                #转成numpy
                div_label_list=np.array(div_label_list)
                div_list=np.array(div_list)
                label_list=np.array(label_list)
                pred_list=np.array(pred_list)
                
                #除去0电位部分数据索引
                good_pos=np.where(div_label_list!=0)
                div_label_list=div_label_list[good_pos]
                div_list=div_list[good_pos]

                label_list=label_list[good_pos]
                pred_list=pred_list[good_pos]

                #如果阈值相差在0.01mv之内，则两个标签相同
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
                print(f1_score(label_list, pred_list, average=None))
                
                # 将类别0、1、2的数量写入文件，以及precision和recall和f1-score
                file.write(f"测试集情况：\n")
                file.write(f"类别0数量: {len(np.where(label_list==0)[0])} | 类别1数量: {len(np.where(label_list==1)[0])} | 类别2数量: {len(np.where(label_list==2)[0])}\n")
                file.write(f"类别0的precision: {precision_score(label_list, pred_list, average=None)[0]} | 类别1的precision: {precision_score(label_list, pred_list, average=None)[1]} | 类别2的precision: {precision_score(label_list, pred_list, average=None)[2]}\n")
                file.write(f"类别0的recall: {recall_score(label_list, pred_list, average=None)[0]} | 类别1的recall: {recall_score(label_list, pred_list, average=None)[1]} | 类别2的recall: {recall_score(label_list, pred_list, average=None)[2]}\n")
                file.write(f"类别0的f1-score: {f1_score(label_list, pred_list, average=None)[0]} | 类别1的f1-score: {f1_score(label_list, pred_list, average=None)[1]} | 类别2的f1-score: {f1_score(label_list, pred_list, average=None)[2]}\n")
                # 缓冲区刷新
                file.flush()




                # 打印每个epoch的验证损失
                mean_val_loss = total_val_loss / len(val_loader)
                print(f"Fold: {fold} | Epoch: {epoch+1} | Mean Validation Loss: {mean_val_loss:.6f}")                  

                # 将损失值追加到文件中
                file.write(f"Epoch: {epoch+1} | 训练损失: {mean_train_loss:.6f} | 验证损失: {mean_val_loss:.6f}\n")
                file.flush()  # 刷新缓冲区，确保内容被写入文件

                #损失加入列表中
                train_loss_records[fold-1].append(mean_train_loss)
                val_loss_records[fold-1].append(total_val_loss / len(val_loader))

                # 保存模型
                save_name = os.path.join(save_dir, "model_{}_epoch{}.pth".format(fold, epoch+1))
                torch.save(model.state_dict(), save_name)

                # 检查验证损失是否有改善
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1

                # 判断是否进行过早停止
                if no_improvement > patience:
                    print("Validation loss has not improved for 5 epochs!")
                    # print("Early stopping!")
                    # break