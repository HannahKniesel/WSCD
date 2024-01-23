from tkinter import Y
import torch
import wandb
import numpy as np

from Utils import *
from Variables import *

############################
# Optimizers               #
############################
# python Main_Binary.py --classifier_path "./TrainingRuns/Herpes/Binary/Classifier_Ablations/42_Classifier_Time_-1_wd8uuu30/" --data_split "val" --project Debug --max_iters 50 --init_cam false

class TranslationMatrix_iterative(torch.nn.Module):
    def __init__(self,init_pos = None, gaussian_pdf = False):
        super().__init__()
        if(init_pos != None):
            self.translation = torch.nn.Parameter(init_pos)
        else: 
            pos = 2*torch.rand(2,) - 1
            self.translation = torch.nn.Parameter(pos)
        Y, X = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
        Y = 2*(Y/IMG_SIZE[1])-1
        X = 2*(X/IMG_SIZE[0])-1
        self.register_buffer("x_grid", torch.from_numpy(X.astype(np.float32)))
        self.register_buffer("y_grid", torch.from_numpy(Y.astype(np.float32)))
        self.register_buffer("pi", torch.tensor([np.pi]))
        self.register_buffer("gaussian_pdf", torch.tensor([gaussian_pdf]))



    def forward(self, std, radius): #, input):
        distances = torch.sqrt((self.x_grid - self.translation[0])**2 + (self.y_grid-self.translation[1])**2  + 1e-8)
        if(std == 0):
            masks = (distances<((radius/IMG_SIZE[0])*2)).float()   
        else:
            fac = 1
            if(self.gaussian_pdf):
                fac = (1/(std*torch.sqrt(2*self.pi)))
            masks = fac *torch.exp(-torch.pow(distances, 2.) / (2 * torch.pow(std,2.)))
        masks = masks.view(1,1,IMG_SIZE[0],IMG_SIZE[1])#[None,None,...]
        return masks

    def get_pixel_translation(self):
        pos = self.translation.detach().cpu().numpy()
        pos = (pos+1)/2
        pos = np.clip(pos,0,1)*IMG_SIZE
        return pos.astype(np.int16)

