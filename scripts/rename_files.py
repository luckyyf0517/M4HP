import os
import shutil

if __name__ == '__main__': 
    dir_name = '/remote-home/iot_yanyifan/mmwave_dataset_workspace/collectedData'
    folder_list  = sorted(os.listdir(dir_name))
    folder_list = [os.path.join(dir_name, folder) for folder in folder_list]
    
    for i, folder_name in enumerate(folder_list): 
        new_name = os.path.join(dir_name, 'seq_%04d' % (i + 1))
        os.rename(folder_name, new_name)
        print('rename:', folder_name, '->', new_name)