import sys
# sys.path.insert(0,'../')

import argparse

from Detector.FasterRCNN import Detection_FRCNN
from Utils import *


if __name__ == "__main__":

    print("******************************")
    print("MINIMAL LABELS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Minimal Labels')

    parser.add_argument('--dataset', type = str, default="herpes", choices=["herpes", "adeno", "noro", "papilloma", "rota"], help='which dataset to use (default:herpes)')
    
    #Training Parameters
    parser.add_argument('--frcnn_bs', type = int, default=16, help='Batch size (default:2)')
    parser.add_argument('--frcnn_n_iters', type = int, default=1000000, help='Maximum iterations (default:1000000)')
    parser.add_argument('--frcnn_lr', type = float, default=0.0001, help='Learning rate for Faster RCNN (default: 0.0001)')
    
    parser.add_argument('--log_path', type = str, default="./TrainingRuns/", help='Logging directory (default: ./TrainingRuns/)')
    parser.add_argument('--project', type = str, default="WSCD", help='wandb project (default:Debug)')
    parser.add_argument('--wandb_mode', type = str, default="online", choices=["online", "offline"], help='wandb mode (default:offline)')
    parser.add_argument('--num_img', type = float, default=1.0, help='For debugging: percentage of training data to use (default: 1.)')
    parser.add_argument('--preload', type = str, default="false", choices=["false", "true"], help='preload data (default:true)')
    parser.add_argument('--backbone', type = str, default="resnet101", choices=["resnet50", "resnet101"], help='backbone of FasterRCNN (default:resnet101)')
    parser.add_argument('--annotation_time', type = int, default=-1, help='Annotation times in seconds for dataset annotation (default:-1)')
    parser.add_argument('--percentage', type = float, default=-1, help='Amount of data to use (default:-1)')
    parser.add_argument('--seeds', type = int, nargs="+", default=[42], help='Multiple seeds for multiple datasplits possible (default:42)')
    

    args = parser.parse_args()
    args.preload = bool(args.preload == "true")

    args.log_path = f"{args.log_path}/{args.dataset}/Location/"

    print("Parameters:")
    print(args)
    deterministic() 
    
    for seed in args.seeds:
        f = Detection_FRCNN(args, LOCATION, seed, LOCATION_TIMINGS)
        f.train()
