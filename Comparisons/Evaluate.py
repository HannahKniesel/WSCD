import sys
sys.path.insert(0, '..')

import argparse
from Variables import *
from Utils import *
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pathlib import Path
import logging
import numpy as np
import os
import wandb


def compute_BB_from_saliency(gray_cam, score, percentage, capsideradius, use_size, size_std): #, center_of_mass): 
    gray_cam = gray_cam.squeeze().numpy()
    t = percentage*np.max(gray_cam)
    bin_cam = (gray_cam>t).astype(np.uint8)*255
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_cam, 8, cv2.CV_32S)
    box = np.array([]).reshape(-1, 4)
    box = torch.as_tensor(box, dtype=torch.float32)
    preds_metric = [dict(boxes=box, scores=torch.tensor([]), labels=torch.tensor([], dtype=torch.int64))]

    if(numLabels>1): # detected virus
        bboxes = []
        bb_scores = []
        for label_idx in range(numLabels):
            centroid = centroids[label_idx]
            label = (labels == label_idx)
            if(np.max(gray_cam[label]) < t): # background
                continue

            x = np.argwhere(np.max(label, axis = 0))
            y = np.argwhere(np.max(label, axis = 1))
            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()
            w = xmax - xmin
            h = ymax - ymin 
            if(size_std>0):
                max_size = (1+size_std)*(capsideradius*2)
                min_size = (1-size_std)*(capsideradius*2)

                # box size does not fit --> discard
                if((w<min_size) or (h<min_size) or (w>max_size) or (h>max_size)):
                    continue
            
            if(use_size):
                x,y = centroid
                xmin = np.max((x - capsideradius, 0))
                ymin = np.max((y - capsideradius, 0))
                xmax = np.min((x + capsideradius, 224))
                ymax = np.min((y + capsideradius, 224))
                w = xmax - xmin
                h = ymax - ymin 
            
            #bb has no shape
            if((w<=1) or (h<=1)):
                continue
            
            bboxes.append([xmin,ymin,xmax,ymax])
            max_act = 1 
            bb_scores.append(score*max_act) 
        preds_metric = [dict(boxes=torch.tensor(bboxes).type(torch.float), scores=torch.as_tensor(bb_scores), labels=torch.tensor([1.], dtype=torch.int64).repeat(len(bboxes)))]
    return preds_metric

def qualitative_result(img, gray_cam, prediction, target, save_path, log_to_wandb = False):
    fig, ax = plt.subplots()
    ax.imshow(min_max(img.squeeze().permute(1,2,0)))
    ax.imshow(gray_cam.squeeze(), alpha=0.3, cmap="plasma") #"inferno"

    # add GT boxes
    for bbox in target[0]['boxes']:
        xmin,ymin,xmax,ymax = bbox
        rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # add predicted boxes + their score
    for bbox, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        xmin,ymin,xmax,ymax = bbox
        x = ((xmax-xmin)/2) + xmin
        y = ((ymax-ymin)/2) + ymin

        rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(x, y, c="red")
        s_percent = int(score*100)
        pad = 5
        ax.text(xmin+2, ymin-2, str(s_percent)+"%", 
        style='normal', color="white", fontsize=30,
        bbox={'facecolor': "red", 'alpha': 0.5, 'pad': pad, 'linewidth': 0}, horizontalalignment="left")

    plt.tight_layout()
    ax.set_axis_off()
    plt.savefig(save_path) 

    if(log_to_wandb and (target[0]['boxes'].shae[0]>0)):
        wandb.log({"Test Image": wandb.Image(plt)})

    plt.close()

def evaluate(dataloader, percentage, size_std = -1, use_size = False, save_path=None):
    for img,gray_cam,score,gt_boxes,radius,name in dataloader:

        # get GT for metric 
        label = torch.tensor([])
        num_boxes = gt_boxes[0].shape[0]
        if(num_boxes>0):
            label = torch.tensor([1.], dtype=torch.int64).repeat(num_boxes)
        targets_metric = [dict(boxes=gt_boxes[0].type(torch.float), labels=label)]

        # when score == 0, there are no BB detected
        if(score == 0):
            bboxes = np.array([]).reshape(-1, 4)
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            preds_metric = [dict(boxes=bboxes, scores=torch.tensor([]), labels=torch.tensor([], dtype=torch.int64))]
        # compute predicted boxes from saliency
        else: 
            preds_metric = compute_BB_from_saliency(gray_cam, score, percentage, radius, use_size, size_std) #, args.use_center_of_mass)
        result = metric(preds_metric, targets_metric)

        if(save_path):
            qualitative_result(img, gray_cam, preds_metric, targets_metric, f"{save_path}/{name}.jpg", args.log_img_to_wandb)

    result = metric.compute()
    return result


class Saliency(Dataset):
    def __init__(self, path):
        self.paths = glob.glob(path+"*.pkl")
        pass
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img, gray_cam, score, gt_boxes, radius = read_pickle(path)
        return img,gray_cam,score,gt_boxes,radius,Path(path).stem


if __name__ == "__main__":

    print("******************************")
    print("Comparison")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Comparison')

    # General Parameters
    parser.add_argument('--test_data', type = str, help='Path to test data')
    parser.add_argument('--val_data', type = str, default="", help='Path to val data')
    parser.add_argument('--best_t', type = float, default=0.15, help='Threshold for binary mask')
    parser.add_argument('--use_size', type = str, default="false", choices=["true", "false"], help='Use size')
    parser.add_argument('--filter_boxes', type = str, default="false", choices=["true", "false"], help='Filter boxes bsed on virus size')
    parser.add_argument('--filter_boxes_std', type = float, default=-1, help='Size range of boxes to filter (boxes in range [capside_size - filter_boxes_std, capside_size + filter_boxes_std]), If negative, compute on validation set')
    parser.add_argument('--group', type = str, default="default", help='wandb group name (should contain method + dataset + filter_size)')
    parser.add_argument('--log_img_to_wandb', type = str, default="false", choices=["true", "false"], help='Filter boxes bsed on virus size')




    args = parser.parse_args()
    args.use_size = bool(args.use_size == "true")
    args.filter_boxes = bool(args.filter_boxes == "true")
    args.log_img_to_wandb = bool(args.log_img_to_wandb == "true")


    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    save_path = f"{Path(args.test_data).parent}/logs_{date_time}/"
    img_path_fixedsize = save_path+"/Images_fixedsize/"
    img_path = save_path+"/Images/"
    os.makedirs(img_path, exist_ok = True)
    os.makedirs(img_path_fixedsize, exist_ok = True)
    logging.basicConfig(filename=save_path+'/log.log', level=logging.DEBUG)

    os.environ['WANDB_PROJECT']= "Comparisons"
    wandb.init(config = args, reinit=True, group = args.group, mode = "online")
    wandb.run.name = date_time

    # validation 
    if(args.val_data != ""): 
        ds_val = Saliency(args.val_data) 
        data_loader_val = torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)

        # find best threshold and capsidsize range
        if((args.best_t < 0) and (args.filter_boxes_std < 0) and args.filter_boxes):
            print("VALIDATION::Find best threshold and capsidsize range on validation set")
            logging.info("VALIDATION::Find best threshold and capsidesize range on validation set")

            best_t = 0
            best_std = 0
            best_map50 = 0
            for percentage in np.arange(0,1,0.05):
                for size_std in np.arange(0,1,0.05):
                    metric = MeanAveragePrecision() 
                    result = evaluate(data_loader_val, percentage, size_std=size_std, save_path=None)

                    if(result['map_50']>best_map50):
                        best_map50 = result['map_50']
                        best_t = percentage
                        best_std = size_std
                    print(f"VALIDATION:: mAP50 = {result['map_50']} for t={percentage} and size_std = {size_std}")
                    logging.info(f"VALIDATION:: mAP50 = {result['map_50']} for t={percentage} and size_std = {size_std}")
                    wandb.log({"Validation/Current Best": best_map50})

        
        # find best threshold
        elif(args.best_t < 0):
            # set best_std variable
            if(args.filter_boxes):
                best_std = args.filter_boxes_std
            else:
                best_std = -1
            print(f"VALIDATION::Find best threshold on validation set. Filter boxes = {args.filter_boxes} with size_std = {best_std}.")
            logging.info(f"VALIDATION::Find best threshold on validation set. Filter boxes = {args.filter_boxes} with size_std = {best_std}.")
          
            best_t = 0
            best_map50 = 0
            for percentage in np.arange(0,1,0.05):
                metric = MeanAveragePrecision() 
                result = evaluate(data_loader_val, percentage,  size_std = best_std, save_path=None)

                if(result['map_50']>best_map50):
                    best_map50 = result['map_50']
                    best_t = percentage
                print(f"VALIDATION:: mAP50 = {result['map_50']} for t={percentage}")
                logging.info(f"VALIDATION:: mAP50 = {result['map_50']} for t={percentage}")
                wandb.log({"Validation/Current Best": best_map50})

        # get best capsidsizerange
        elif((args.filter_boxes_std < 0) and args.filter_boxes):
            #set best_t variable
            best_t = args.best_t
            print(f"VALIDATION::Find best size std on validation set. Use threshold = {args.best_t}.")
            logging.info(f"VALIDATION::Find best size std on validation set. Use threshold = {args.best_t}.")

            best_std = 0
            best_map50 = 0
            for size_std in np.arange(0,1,0.05):
                metric = MeanAveragePrecision() 
                result = evaluate(data_loader_val, best_t, size_std=size_std, save_path=None)

                if(result['map_50']>best_map50):
                    best_map50 = result['map_50']
                    best_std = size_std

                print(f"VALIDATION:: mAP50 = {result['map_50']} for size_std={size_std}")
                logging.info(f"VALIDATION:: mAP50 = {result['map_50']} for size_std={size_std}")
                wandb.log({"Validation/Current Best": best_map50})


        else:
            metric = MeanAveragePrecision() 
            best_t = args.best_t
            best_std = args.filter_boxes_std
            print(f"VALIDATION:: Validation with threshold t = {args.best_t}")
            logging.info(f"VALIDATION:: Validation with threshold t = {args.best_t}")
            result = evaluate(data_loader_val, best_t, best_std, save_path=None)
            best_map50 = result['map_50']
            print(f"VALIDATION:: mAP50 = {result['map_50']} for fixed size_std={best_std} with t={best_t}")
            print(f"VALIDATION:: mAP50 = {result['map_50']} for fixed size_std={best_std} with t={best_t}")
            wandb.log({"Validation/Current Best": best_map50})

        print(f"VALIDATION:: best mAP50 = {best_map50} for t={best_t} and size_std = {best_std}\n")
        logging.info(f"VALIDATION:: best mAP50 = {best_map50} for t={best_t} and size_std = {best_std}\n")
        wandb.log({"Validation/Best Threshold": best_t})
        wandb.log({"Validation/Best Size": best_std})
        wandb.log({"Validation/Best Result": best_map50})


        


    else: 
        best_t = args.best_t
        best_std = args.filter_boxes_std



    # test
    ds_test = Saliency(args.test_data) 
    data_loader_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0) 

    # virus size from saliency map
    metric = MeanAveragePrecision() 
    result = evaluate(data_loader_test, best_t, size_std=best_std, save_path=img_path)
    print(f"TEST:: mAP50 = {result['map_50']} for t={best_t}")
    print(f"TEST:: Results saved to {img_path}")
    logging.info(f"TEST:: mAP50 = {result['map_50']} for t={best_t}")
    wandb.log({"Test/Best Size": best_std})
    wandb.log({"Test/Best Threshold": best_t})
    wandb.log({"Test/Best Result": result['map_50']})

    # fix virus size
    metric = MeanAveragePrecision() 
    result = evaluate(data_loader_test, best_t, size_std=best_std, save_path=img_path_fixedsize, use_size=True)
    print(f"TEST with size:: mAP50 = {result['map_50']} for t={best_t}")
    print(f"TEST with size:: Results saved to {img_path}")
    logging.info(f"TEST with size:: mAP50 = {result['map_50']} for t={best_t}")
    wandb.log({"Test with size/Best Size": best_std})
    wandb.log({"Test with size/Best Threshold": best_t})
    wandb.log({"Testwith size/Best Result": result['map_50']})