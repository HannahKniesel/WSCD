

import sys
sys.path.insert(0, './Detector/')
sys.path.insert(0, './Classifier/')

# sys.path.insert(0,'../')

import argparse
import os

from Classifier.TrainClassifier import TrainingClassifier
from PseudoLabels.Optimize_iterative import OptimizerIter
from PseudoLabels.Optimize_sliding import OptimizerSliding

from Detector.FasterRCNN import Detection_FRCNN

from Utils_Eval import *
from Utils import *



if __name__ == "__main__":

    print("******************************")
    print("BINARY")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Binary')

    # General Parameters
    parser.add_argument('--log_path', type = str, default="./TrainingRuns/", help='Logging directory (default: ./TrainingRuns/)')
    parser.add_argument('--project', type = str, default="WSCD", help='wandb project (default:WSCD)')
    parser.add_argument('--wandb_mode', type = str, default="online", choices=["online", "offline"], help='wandb mode (default:offline)')
    parser.add_argument('--preload', type = str, default="false", choices=["false", "true"], help='preload data (default:false)')
    parser.add_argument('--start_idx', type = int, default=0, help='For debugging: start image of data generation (default:0)')
    parser.add_argument('--dataset', type = str, default="herpes", choices=["herpes", "adeno", "noro", "papilloma", "rota"], help='which dataset to use (default:herpes)')
    parser.add_argument('--num_img', type = float, default=1.0, help='For debugging: percentage of training data to use (default: 1.)')
    parser.add_argument('--backbone', type = str, default="resnet101", choices=["resnet50", "resnet101", "vit"], help='backbone (default:resnet101)')
    parser.add_argument('--annotation_time', type = int, default=-1, help='Annotation times in seconds for dataset annotation (default:-1)')
    parser.add_argument('--percentage', type = float, default=-1, help='Amount of data to use (default:-1)')
    parser.add_argument('--seeds', type = int, nargs="+", default=[42], help='Multiple seeds for repeated experiment possible (default:42)')

   
    # Classifier Training Parameters
    parser.add_argument('--classifier_path', type = str, default='', help='path for pretrained classifier (default: "")')
    parser.add_argument('--classifier_iters', type = int, default=500000, help='maximum number of iterations (default: 500000)')
    parser.add_argument('--classifier_bs', type = int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--classifier_lr', type = float, default=0.00005, help='learning rate (default: 0.00005)')
    parser.add_argument('--classifier_optim', type = str, default='adam',choices=['sgd', 'adam'], help='optimizer (default: adam)')
    parser.add_argument('--classifier_use_amp', type = str, default='true',choices=['false', 'true'], help='use 16bit precision training (default: true)')
    parser.add_argument('--classifier_only', type = str, default='false',choices=['false', 'true'], help='only train classifier (default: false)')


    # Pseudolabel Generation Parameters    
    parser.add_argument('--pseudolabel_kind', nargs="+", type = str, default=['iterative'], choices=['sliding', 'iterative'], help='path for pseudolabels (default: "")')
    parser.add_argument('--std_end', type = float, default=0.5, help='std at last iteration (default: 0.5)')
    parser.add_argument('--std_start', type = float, default=6, help='std at fist iteration (default: 6)')
    parser.add_argument('--min_iters', type = int, default=0, help='Min niters for optim (default:0)')
    parser.add_argument('--scheduler', type = str, default='cos',choices=["cos", "exp", "step", "None"], help='scheduler to use (default: cos)')
    parser.add_argument('--momentum', type = float, default=0.0, help='Momentum term for SGD (default:0.0)')
    parser.add_argument('--val_step', type = int, default=10, help='validation every --val_step iterations (default:10)')
    parser.add_argument('--patience', type = int, default=3, help='Patience for early stopping (default:5)')
    parser.add_argument('--lr_t', type = float, default=0.0005, help='learning rate for translation(default:0.0005)')
    parser.add_argument('--lr_t_final', type = float, default=0.02, help='multiplicative factor for final lr (default:0.02)')
    parser.add_argument('--pseudolabels_use_validation', type = str, default='false',choices=['true', 'false'], help='use validation for pseudolabels (default: false)')
    parser.add_argument('--pseudolabels_gaussian_pdf', type = str, default='true',choices=['true', 'false'], help='Use normalized gaussian (integral = 1) (default: true)')
    parser.add_argument('--selective_search_mode', type = str, default='quality',choices=['fast', 'single', 'quality'], help='Selective search mode (default: fast)')
    parser.add_argument('--selective_search_topn', type = int, default=-1, help='Use top N boxes of selective search (default:-1)')
    parser.add_argument('--nms_max_iou', type = float, default=0.01, help='max iou for NMS (default:0.01)')
    parser.add_argument('--corrupt_size', type = float, default=-1, help='corrupt virus size approximation for label generation (default:-1)')
    parser.add_argument('--pseudolabel_threshold', type = float, default=0.01, help='threshold that virus is still detected. (default:0.01)')
    parser.add_argument('--log_val', type = int, default=0, help='log during validation (default:0)')
    parser.add_argument('--save_data', type = str, default='true',choices=['true', 'false'], help='weather to save data (default: true)')
    parser.add_argument('--pseudolabels_use_amp', type = str, default='true',choices=['true', 'false'], help='use 16bit precision training (default: false)')
    parser.add_argument('--data_split', nargs="+", type = str, default=["test"], choices=["val", "test", "train"], help='what data split to use (default:test)')
    parser.add_argument('--num_virus', type = int, default=-1, help='For debugging: use images with num_virus particles (default:-1= use all)')
    parser.add_argument('--masking', type = str, choices=["mean", "zeros", "inpainting"], default="mean", help='Mask by inpainting or by masking (default: "mean")')
    parser.add_argument('--loss', type = str, default="logit", choices=['score', 'logit', 'oracle'], help='which loss to use (default:score)')
    parser.add_argument('--score_bb', type = str, default='mask_other_virus', choices=['mask_bg', 'mask_other_virus'], help='weather to mask BG or other virus for BB score computation (default: mask_other_virus)')
    parser.add_argument('--initialize', type = str, choices=["gradcam", "random", "selectivesearch", "oracle"], default="gradcam", help='Initialization of position (default: "gradcam")')
    parser.add_argument('--max_iters', type = int, default=50, help='Max niters for optim (default:50)')
    parser.add_argument('--step', type = float, default=0.5, help='step*radius as step of sliding window (default: 0.5)')


    # Faster RCNN for training on pseudo labels
    parser.add_argument('--frcnn_pseudolabels_path', type = str, default="", help='Path to pseudo training labels (default:"")')
    parser.add_argument('--frcnn_bs', type = int, default=16, help='Batch size (default:2)')
    parser.add_argument('--frcnn_n_iters', type = int, default=1000000, help='Maximum iterations (default:1000000)')
    parser.add_argument('--frcnn_lr', type = float, default=0.0001, help='Learning rate for Faster RCNN (default: 0.0001)')
    parser.add_argument('--frcnn_probabilities', type = str, nargs="+", choices=["true", "false"], default=["true"], help='Use probability for FRCNN training (default: "true")')
    
    args = parser.parse_args()
    args.preload = bool(args.preload == "true")
    args.pseudolabels_use_amp = bool(args.pseudolabels_use_amp == "true")
    args.classifier_use_amp = bool(args.classifier_use_amp == "true")
    args.save_data = bool(args.save_data == "true")
    args.classifier_only = bool(args.classifier_only == "true")
    args.pseudolabels_use_validation = bool(args.pseudolabels_use_validation == "true")
    args.pseudolabels_gaussian_pdf = bool(args.pseudolabels_gaussian_pdf == "true")

    args.log_path = f"{args.log_path}/{args.dataset}/Binary/"

    if(args.pseudolabel_threshold < 0): 
        args.pseudolabel_threshold = None

    print("Parameters:")
    print(args)
    deterministic()

    init_classifier_path = args.classifier_path
    
    for i,seed in enumerate(args.seeds):
        
        # Classifier
        if(args.classifier_path == ""):
            print("Start Classifier Training...")
            classifier = TrainingClassifier(args, seed = seed)
            model, best_t, data_paths, classifier_path = classifier.training()
            args.classifier_path = classifier_path
        
        print("Loading classifier from path ("+str(args.classifier_path+"/training_state.pth)"))
        model, best_t, training_data_paths = load_classifier(args.classifier_path+"/training_state.pth", torch.nn.Identity(), args.loss)
        gradcam_model,_,_ = load_classifier(args.classifier_path+"/training_state.pth", torch.nn.Identity(), "score")
        if(args.classifier_only):
            args.classifier_path = init_classifier_path 
            continue
        
        # generate pseudolabels
        if(args.frcnn_pseudolabels_path == ""):
            # Pseudolabels for multiple data splits
            for data_split in args.data_split:
                # when datasplit is training use data paths where the model has been trained for.
                if(data_split == "train"):
                    data_paths = training_data_paths
                else: 
                    data_paths = []
                for pseudolabel_kind in args.pseudolabel_kind:
                    log_path = args.classifier_path+"/Pseudolabels/"+str(data_split)+"/"+str(pseudolabel_kind)+"/" #"/Debug/" 
                    os.makedirs(log_path, exist_ok=True)
                    if(pseudolabel_kind == "iterative"):
                        o = OptimizerIter(args, log_path, data_split, model, gradcam_model, data_paths, seed)
                        path_to_training_labels = o.train()
                    elif(pseudolabel_kind == "sliding"):
                        o = OptimizerSliding(args, log_path, data_split, model, data_paths, seed)
                        path_to_training_labels = o.train()
                    
                    
                    # training labels have been generated --> hence train on them
                    if(path_to_training_labels != None):
                        path_to_training_labels = glob.glob(path_to_training_labels+"/*.pkl")
                        for frcnn_probabilities in args.frcnn_probabilities:
                            if(frcnn_probabilities == "true"):
                                threshold = -1
                            elif(frcnn_probabilities == "false"):
                                threshold = best_t
                            f = Detection_FRCNN(args, CLASSIFICATION, seed, CLASSIFICATION_TIMINGS, path_to_training_labels= path_to_training_labels, threshold = threshold)
                            f.train()
        # pseudolabels already have been generated --> train FRCNN on those
        else: 
            path_to_training_labels = glob.glob(args.frcnn_pseudolabels_path+"/*.pkl")
            f = Detection_FRCNN(args, CLASSIFICATION, seed, CLASSIFICATION_TIMINGS, path_to_training_labels= path_to_training_labels)
            f.train()

        args.classifier_path = init_classifier_path 


        


    
    

    
               
       