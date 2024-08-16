import yaml
import argparse
# from tools import Runner
from collections import namedtuple


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def parse_arg(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--version', type=str, default='test', metavar='B',
                        help='directory of saving/loading')
    parser.add_argument('--config', type=str, default='mscsa_prgcn_demo.yaml', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--gpu', default=[0], type=eval, help='IDs of GPUs to use')                        
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('-sr', '--sampling_ratio', type=int, default=1, help='sampling ratio for training/test (default: 1)')
    parser.add_argument('--keypoints', action='store_true', help='print out the APs of all keypoints')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arg()
   
    with open('./config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    trigger = Runner(args, cfg)
    vis = False if args.vis_dir == 'none' else True
    if args.eval:
        trigger.loadModelWeight('model_best')
        trigger.eval(visualization=vis)
    else:
        trigger.loadModelWeight('model_best')
        trigger.train()