import torch
from torch.utils.data import Dataset,dataloader
from itertools import cycle
from tqdm import tqdm
import pickle
import numpy as np

def custom_collate_fn(batch):
    # 将 batch 中的元素拆分为不同的变量
    waves, labels, rpos, resamples = zip(*batch)

    # 转换波形数据和标签数据为张量
    waves = torch.stack(waves)
    labels = torch.stack(labels)

    # 返回处理后的批数据
    return waves, labels, rpos, resamples

class MyDataset(Dataset):
    def __init__(self, data_path_list, transform=None, st=False):
        self.wave      = []
        self.label     = []
        self.st        = st
        self.transform = transform

        if st:
            self.rpos     = []
            self.resample = []
        
        for data_path in data_path_list:
            # data = pickle.load(open(data_path, 'rb'))
            if not st:
                data    = pickle.load(open(data_path, 'rb'))
                x_train = np.array(data['wave']) #每一条ecg数据有256个点，x_train.shape = (30, 256)
                y_train = np.array(data['label'] ,dtype=object)
                
                self.wave.extend(x_train)
                self.label.extend(y_train)

            else:
                data = pickle.load(open(data_path, 'rb'))
                self.wave.extend(data['wave'])
                self.label.extend(data['label'])
                self.rpos.extend(data['rpos'])
                self.resample.extend(data['resample'])

        print(np.array(self.label).shape)
        print(np.array(self.wave).shape)
        if st:
            print(np.array(self.rpos).shape)
            print(np.array(self.resample).shape)

    def __len__(self):
        return len(self.wave)
    
    def __getitem__(self, item):

        if not self.st:

            tmp_wave  = self.wave[item].astype(np.float32)
            tmp_label = self.label[item].astype(np.float32)

            
            # 对一维信号进行数据增强
            if self.transform is not None:
                tmp_wave = self.transform(tmp_wave)

            return torch.Tensor(tmp_wave), torch.Tensor(tmp_label)

        else:
            
            this_wave     = self.wave[item].astype(np.float32)
            this_label    = self.label[item]
            this_rpos     = self.rpos[item]
            this_resample = self.resample[item].astype(np.float32)

            # 维度转换
            this_wave     = this_wave.reshape((1, 256))
            this_label    = this_label.reshape((1, ))
            this_rpos     = this_rpos.reshape((1, ))
            this_resample = this_resample.reshape((1,))

            # # 查看哪个数据不是sequence
            # print(this_wave.shape)
            # print(this_label.shape)
            # print(this_rpos.shape)
            # print(this_resample.shape)
            
            return torch.Tensor(this_wave), torch.Tensor(this_label), this_rpos, torch.Tensor(this_resample)
    

if __name__ == "__main__":
    # test_data_path = '//192.168.2.8/xjk/Algorithm/ECG_ST/transformer-data/训练数据/ltst_result_segment_norm/ltst_part_s200.data'
    test_data_path = "st_test_1points\European_ST_1_single.data"
    test_dataset = MyDataset([test_data_path], st=True)

    # test_data_path = "data_adjust\ltst_part_s200_single.data"
    # test_dataset = MyDataset([test_data_path], st=False)

    print("done")
    train_loader = dataloader.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
    print("finish")




