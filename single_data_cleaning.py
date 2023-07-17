import pickle
import os
import logging
from datetime import datetime
import numpy as np

path = "data"
data_list = os.listdir(path)
save_dir = "data_cleaning"
os.makedirs(save_dir, exist_ok=True)

# 配置日志记录
script_name = os.path.basename(__file__)
timestamp   = datetime.now().strftime("%Y%m%d%H%M%S")
filename    = "logs/{}_{}.log".format(script_name, timestamp)

for name in data_list:
    data_path = os.path.join(path, name)
    logging.info("processing {}".format(name))
    data = pickle.load(open((data_path), 'rb'))
    x_train = np.array(data['wave'])
    y_train = np.array(data['label'], dtype=object)
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
    save_name = name.split('.')[0] + '.data'
    save_path = os.path.join(save_dir, save_name)
    pickle.dump(data, open(save_path, 'wb'))
    logging.info("save to {}".format(save_path))





