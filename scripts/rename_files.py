import os
import glob
import shutil
import pprint

if __name__ == '__main__': 
    dir_name = '/remote-home/iot_yanyifan/mmwave_dataset_workspace/collectedData'
    folder_list = sorted(glob.glob(os.path.join(dir_name, 'seq_04*')))
    # folder_list  = sorted(os.listdir(dir_name))
    folder_list = [os.path.join(dir_name, folder) for folder in folder_list]
    
    for i, folder_name in enumerate(folder_list): 
        # new_name = os.path.join(dir_name, 'seq_%04d' % (400 + i + 1))
        if i < 27: 
            new_name = os.path.join(dir_name, 'P07A%02d' % (i))
        else: 
            new_name = os.path.join(dir_name, 'P07R%02d' % (i - 27))
        os.rename(folder_name, new_name)
        print('rename:', folder_name, '->', new_name)