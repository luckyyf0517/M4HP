import time
import torch
import numpy as np

data_path = '/remote-home/iot_yanyifan/mmwave_dataset_workspace/collectedData/seq_0001/mmwave/adc_data_hori.bin'
tic = time.time()
with open(data_path, 'rb') as f: 
    # data = torch.tensor()
    data = np.fromfile(data_path, dtype=np.int16)
    data = torch.from_numpy(data)
print(time.time() - tic)

data_path = '/root/raw_data/demo/seq_0001/mmwave/adc_data_hori.bin'
tic = time.time()
with open(data_path, 'rb') as f: 
    # data = torch.tensor()
    data = np.fromfile(data_path, dtype=np.int16)
    data = torch.from_numpy(data)
print(time.time() - tic)