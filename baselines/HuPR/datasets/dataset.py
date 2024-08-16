import os
import random
import json
import time
import yaml
import torch
import numpy as np
import multiprocessing
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from random import sample
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
# from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from misc.coco import COCO
from misc.cocoeval import COCOeval
from datasets.base import BaseDataset, generateGTAnnot


# def getDataset(phase, cfg, args, random=True):
#     return HuPR3D_horivert(phase, cfg, args, random)

    
class HuPR3D_raw(BaseDataset):
    def __init__(self, phase, cfg, args, shuffle=False):
        if phase not in ('train', 'val', 'test'):
            raise ValueError('Invalid phase: {}'.format(phase))
        super(HuPR3D_raw, self).__init__(phase)
        self.numFrames = cfg.DATASET.numFrames
        self.num_group_frames = cfg.DATASET.num_group_frames
        self.numChirps = cfg.DATASET.numChirps
        self.r = cfg.DATASET.rangeSize
        self.w = cfg.DATASET.azimuthSize
        self.h = cfg.DATASET.elevationSize
        self.num_keypoints = cfg.DATASET.num_keypoints
        self.sampling_ratio = args.sampling_ratio
        self.dirRoot = cfg.DATASET.dataDir
        self.idx_to_joints = cfg.DATASET.idx_to_joints
        self.random = shuffle
        self.cfg = cfg

        # To avoid json loading fail
        while True: 
            try: 
                generateGTAnnot(cfg, phase)
                self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % phase)
                self.coco = COCO(self.gtFile)
                break
            except Exception as e: 
                print('Error in generating GT, retrying...', e)
                
        self.imageIds = self.coco.getImgIds()
        
        self.keyimageIds = [self.imageIds[i] for i in range(0, len(self.imageIds), self.sampling_ratio)]
        
        self.length_map = {}
        for name in self.imageIds: 
            imageId = '%09d' % name
            seq_name = imageId[: 4]
            if seq_name not in self.length_map: 
                self.length_map[seq_name] = 1
            else: 
                self.length_map[seq_name] += 1
        
        self.VRDAEPaths_hori = []
        self.VRDAEPaths_vert = []
        
        self.current_seq = None
        self.current_hori_npy = None
        self.current_vert_npy = None
        
        self.annots = self._load_coco_keypoint_annotations()
        self.phase = phase
        
        self.loaded_bins = {}

    def evaluateEach(self, loadDir):
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        keypoint_list = []
        for i in range(self.num_keypoints):
            coco_eval.evaluate(i)
            coco_eval.accumulate()
            coco_eval.summarize()
            info_str = []
            for ind, name in enumerate(stats_names):
                info_str.append((name, coco_eval.stats[ind]))
            keypoint_list.append(info_str[0][1])
        # for i in range(self.num_keypoints):
        #     print('%s: %.3f' % (self.idx_to_joints[i], keypoint_list[i]))
        return keypoint_list # return the value of AP
    
    def evaluate(self, loadDir):
        res_file = os.path.join(loadDir, "%s_results.json"% self.phase)
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75']#, 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = {}
        for ind, name in enumerate(stats_names):
            info_str[name] = coco_eval.stats[ind]
        
        # for idx_metric in range(10):
        #     print("%s:\t%.3f\t"%(info_str[idx_metric][0], info_str[idx_metric][1]), end='')
        #     if (idx_metric+1) % 5 == 0:
        #         print()
        return info_str # return the value of AP

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.imageIds:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)
        rec = []
        for obj in objs:
            joints_2d = np.zeros((self.num_keypoints, 2), dtype=np.float64)
            joints_2d_vis = np.zeros((self.num_keypoints, 2), dtype=np.float64)
            for ipt in range(self.num_keypoints):
                joints_2d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_2d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_2d_vis[ipt, 0] = t_vis
                joints_2d_vis[ipt, 1] = t_vis
            rec.append({
                'joints': joints_2d,
                'joints_vis': joints_2d_vis,
                'bbox': obj['bbox'], # x, y, w, h
                'imageId': obj['image_id']
            })
        return rec

    def _load_bin(self, seq_name, duration=None): 
        # only load once into memory
        bin_dir = os.path.join('/root/raw_data/demo/', 'seq_' + seq_name, 'mmwave')
        
        with open(os.path.join(bin_dir, "radar_config.yaml"), 'r') as f:
            mmwave_cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        if not seq_name in self.loaded_bins: 
            # loading raw data into memory is useful to speed up
            print(multiprocessing.current_process().name, 'Rank', dist.get_rank(), 'reloading', seq_name, f'(Already loaded: {len(self.loaded_bins)})'); tic = time.time()
            bin_hori = torch.from_numpy(np.fromfile(open(os.path.join(bin_dir, 'adc_data_hori.bin'), 'rb'), dtype=np.int16))
            bin_vert = torch.from_numpy(np.fromfile(open(os.path.join(bin_dir, 'adc_data_vert.bin'), 'rb'), dtype=np.int16))
            bin_hori = self._reshape_bin(bin_hori, mmwave_cfg)
            bin_vert = self._reshape_bin(bin_vert, mmwave_cfg)
            self.loaded_bins[seq_name] = {'hori': bin_hori, 'vert': bin_vert}
            # print(multiprocessing.current_process().name, 'reload cost', time.time() - tic)
        else: 
            bin_hori = self.loaded_bins[seq_name]['hori']
            bin_vert = self.loaded_bins[seq_name]['vert']
        
        if duration is not None: 
            assert duration == mmwave_cfg['num_frames']
        
        return bin_hori, bin_vert, mmwave_cfg
        
    def _reshape_bin(self, bin_file, mmwave_cfg): 
        adc_shape = torch.tensor(mmwave_cfg['adc_shape'])
        file_size = torch.prod(adc_shape) * 2
        read_file_size = bin_file.shape[0]
        
        bin_file = F.pad(bin_file, (0, file_size - read_file_size), 'constant', 0)
        lvds0 = bin_file[0::2]
        lvds1 = bin_file[1::2]
        lvds0 = lvds0.reshape(*adc_shape)
        lvds1 = lvds1.reshape(*adc_shape)
        return lvds1 + 1j * lvds0
        
    def __getitem__(self, index):
        # sampling
        index = index * self.sampling_ratio
        
        # get frameId and seqName
        imageId = self.imageIds[index]
        imageId = '%09d' % imageId
        seqName = imageId[: 4]
        frameId = imageId[-4: ]
        
        tic = time.time()        
        duration = self.length_map[seqName]
        startFrame = int(frameId) - self.num_group_frames//2 
        endFrame = int(frameId) + self.num_group_frames//2
        if startFrame < 0: 
            endFrame -= startFrame
            startFrame = 0
        if endFrame > duration - 1: 
            startFrame -= (endFrame - duration + 1)
            endFrame = duration - 1
        
        VRDAERealImag_horis, VRDAERealImag_verts, mmwave_cfg = self._load_bin(seqName, duration)
        VRDAERealImag_hori = VRDAERealImag_horis[startFrame: endFrame]
        VRDAERealImag_vert = VRDAERealImag_verts[startFrame: endFrame]
        
        joints = torch.LongTensor(self.annots[index]['joints'])
        
        # classification demo
        if self.cfg.MODEL.runClassification:
            person_id = int(seqName) // 100 # 1, 2, 3, 4 
            # assert person_id > 0, 'Person ID is not specified'
            if person_id == 0: 
                if 0 < int(seqName) <= 20: 
                    person_id = 5
                elif 20 < int(seqName) <= 40:
                    person_id = 4
                elif 40 < int(seqName) <= 60:
                    person_id = 6
                elif 60 < int(seqName) <= 80:
                    person_id = 7
            action_id = int(seqName) % 100
        else: 
            person_id = 0
            action_id = 0   # (0 denotes not specified action)
        
        person_label = torch.tensor(person_id - 1)
        action_label = torch.tensor(action_id - 1)
        
        item_dict = {
            'mmwave_cfg': mmwave_cfg, 
            'VRDAEmap_hori': VRDAERealImag_hori,
            'VRDAEmap_vert': VRDAERealImag_vert,
            'imageId': imageId,
            'jointsGroup': joints, 
            'person_label': person_label,
            'action_label': action_label,
        }
            
        if self.annots[index]['bbox'] is not None: 
            item_dict['bbox'] = torch.FloatTensor(self.annots[index]['bbox'])
            
        # if self.cfg.runSegmentation: 
        #     item_dict['RAmask'] = torch.from_numpy(np.load(os.path.join(self.dirRoot, seqName, 'range_azimuth_mask', frameId.zfill(9) + '.npy'))) # (64, 64)
        #     item_dict['REmask'] = torch.from_numpy(np.load(os.path.join(self.dirRoot, seqName, 'range_elevate_mask', frameId.zfill(9) + '.npy'))) # (64, 64)
        
        # print(multiprocessing.current_process().name, 'load cost time', time.time() - tic)
        return item_dict
    
    def __len__(self):
        return len(self.imageIds) // self.sampling_ratio
    
 