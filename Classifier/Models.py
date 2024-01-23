
from copy import deepcopy
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
import torch.nn.functional as F
import torch
import numpy as np
from Variables import *

    
class Classifier_ResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, OUTPUT_NEURONS)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # for resnet50 use pretrained weights from Conrad, Ryan, and Kedar Narayan. "CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning." Elife 10 (2021): e65894.
        state = torch.load(EM_PRETRAINED_WEIGHTS, map_location='cpu')
        state_dict = state['state_dict']
        #format the parameter names to match torchvision resnet50
        resnet50_state_dict = deepcopy(state_dict)
        for k in list(resnet50_state_dict.keys()):
            #only keep query encoder parameters; discard the fc projection head
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                resnet50_state_dict[k[len("module.encoder_q."):]] = resnet50_state_dict[k]
            #delete renamed or unused k
            del resnet50_state_dict[k]
        # load model weights
        model.load_state_dict(resnet50_state_dict, strict=False)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class Classifier_Oracle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, transformed_mask: torch.Tensor, gt_mask: torch.Tensor, capside_radius: torch.Tensor) -> torch.Tensor:
        area = (np.pi*capside_radius**2).to(DEVICE)
        overlap = transformed_mask.squeeze()*gt_mask.to(DEVICE).squeeze()
        return torch.sum(overlap)/area
    
class Classifier_ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, OUTPUT_NEURONS)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class Classifier_ResNet101(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, OUTPUT_NEURONS)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    


