import argparse
import wandb
import torch
import numpy as np
from copy import deepcopy
import torch

import sys
sys.path.insert(0,'..')
sys.path.insert(0, '../Detector/')
from Utils import *
from Utils_Eval import *
from Variables import *
from .Optimize import Optimizer

import matplotlib.patches as patches



class OptimizerSliding(Optimizer):
    
    def __init__(self, args, log_path, data_split, model, data_paths, seed):
        title="Sliding_"
        super().__init__(args, log_path, data_split, title, model, data_paths, seed)
        
    
    def init_pos_cam(self, cam):        
        _, y,x = np.unravel_index(cam.argmax(), cam.shape)
        plt.close()
        plt.clf()
        plt.figure()
        plt.imshow(cam.squeeze())
        plt.scatter(x,y)
        wandb.log({"CAM/img ": wandb.Image(plt)})
        plt.close()
        x = (2*(x/IMG_SIZE[0]))-1
        y = (2*(y/IMG_SIZE[0]))-1
        return torch.FloatTensor([-x, -y])   

    def train_translation(self, batch, i):
        curr_img = batch['image']       
        positions, avg_number_position = self.train_iter(batch, curr_img)
        positions = np.array(positions)
        if(not np.any(positions == -1)):
            positions = np.clip(positions, 0, IMG_SIZE[0]-1)
            positions, scores, num_virus = self.select_masks_by_iou(positions, batch['capsideradius']*2, batch['image'], self.args.nms_max_iou, individual_masks = True)
            scores = self.recompute_scores(positions, batch['capsideradius'], batch['image'])
        else: 
            scores = np.array([])
            num_virus = 0

        return positions, scores, num_virus, avg_number_position
    
    def plot(self,  image, positions, scores, capside_radius, target_boxes, save_path):
        # Create figure and axes
        fig, ax = plt.subplots()
        ax.imshow(image.squeeze(), cmap="gray")

        for box in target_boxes:
            xmin,ymin,xmax,ymax = box
            rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=2, edgecolor='g', facecolor='none')#, alpha = 0.5)
            ax.add_patch(rect)
        for position,score in zip(positions, scores):
            xmin = np.max((position[0]-capside_radius, 0))
            xmax = np.min((position[0]+capside_radius, IMG_SIZE[0]))
            ymin = np.max((position[1]-capside_radius, 0))
            ymax = np.min((position[1]+capside_radius, IMG_SIZE[0]))
            rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=2, edgecolor='r', facecolor='none', alpha = np.min([0.3 + score,1]))
            ax.add_patch(rect)

        ax.set_axis_off()
        if(save_path):
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def train_iter(self, batch, curr_img):  
        pos_x = np.arange(0, IMG_SIZE[0]+batch['capsideradius'], self.args.step*batch['capsideradius'])
        pos_y = np.arange(0, IMG_SIZE[1]+batch['capsideradius'], self.args.step*batch['capsideradius'])
        x,y = np.meshgrid(pos_x, pos_y)
        positions = np.stack([x,y], axis=-1).reshape(-1,2)
        
        masks = generate_masks_from_positions(positions, batch['capsideradius'].numpy())

        with torch.no_grad():
            all_scores = self.get_scores_batchified(masks, curr_img, self.model, (batch['capsideradius']*2).to(DEVICE))
            all_scores = np.array(all_scores)
            positions = positions[all_scores>self.args.pseudolabel_threshold,:]
            scores = all_scores[all_scores>self.args.pseudolabel_threshold]
            if(positions.shape[0] == 0):
                positions = np.array([-1,-1]).astype(np.int16)    

        return positions, masks.shape[0]
    
                     
