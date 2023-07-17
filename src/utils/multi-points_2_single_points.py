import os
import pickle
import logging
import numpy as np
from datetime import datetime

def create_single_points_data(data_dir, save_dir, st=False):
    '''
    将30拍ecg数据转换为单拍数据，便于后续处理
    '''
    # 配置日志记录
    script_name = os.path.basename(__file__)
    timestamp   = datetime.now().strftime("%Y%m%d%H%M%S")
    filename    = "logs/{}_{}.log".format(script_name, timestamp)
    
    logging.basicConfig(filename=filename, level=logging.INFO)
    os.makedirs(save_dir, exist_ok=True)

    for name in os.listdir(data_dir):
        logging.info("processing {}".format(name))
        data_path = os.path.join(data_dir, name)
        data = pickle.load(open(data_path, 'rb')).tolist()
        # data = pickle.load(open(data_path, 'rb'))
        if not st:
            x_train = data['wave']
            y_train = data['label']
            wave = []
            label = []
            n = len(x_train)
            m = len(x_train[0])

            for i in range(n):
                for j in range(m):
                    start, end = y_train[i][j][-2:]

                    if x_train[i][j][start] - x_train[i][j][end] != 0:
                        wave.append(x_train[i][j])
                        label.append(y_train[i][j])

            data = {'wave': wave, 'label': label}
            save_name = name.split('.')[0] + '_single.data'
            save_path = os.path.join(save_dir, save_name)
            pickle.dump(data, open(save_path, 'wb'))
            logging.info("save to {}".format(save_path))
        
        else:
            x_train    = data['wave']
            y_train    = data['label']
            z_rpos     = data['rpos']
            a_resample = data['resample']

            wave     = []
            label    = []
            rpos     = []
            resample = []
            n        = len(x_train)
            m        = len(x_train[0])

            for i in range(n):
                for j in range(m):
                    wave.append(x_train[i][j])
                    label.append(np.array(y_train[i][j]))
                    rpos.append(np.array(z_rpos[i][j]))
                    resample.append(a_resample[i][j])

            # data = {'wave': wave, 'label': label, 'rpos': rpos, 'resample': resample}
            # 存储的data的value改成numpy格式
            data = {'wave': np.array(wave), 'label': np.array(label), 'rpos': np.array(rpos), 'resample': np.array(resample)}
            save_name = name.split('.')[0] + '_single.data'
            save_path = os.path.join(save_dir, save_name)
            pickle.dump(data, open(save_path, 'wb'))
            logging.info("save to {}".format(save_path))

if __name__ == "__main__":
    data_dir = "st_test_30points"
    save_dir = "st_test_1points"
    create_single_points_data(data_dir, save_dir, st=True)
    print("done")
