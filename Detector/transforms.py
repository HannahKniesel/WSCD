import random
import torch
import numpy as np
from torchvision.transforms import functional as F
import torchvision.transforms


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            image = np.flip(image, 0).copy() #image.flip(-1)
            bbox = target["boxes"]
            if(len(bbox) > 0):
                bbox[:, [1, 3]] = (width - bbox[:, [3, 1]])%(width+1)
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            
            height, width = image.shape[:2]
            image = np.flip(image, 1).copy() #image = image.flip(1)
            bbox = target["boxes"]
            if(len(bbox) > 0):
                bbox[:, [0, 2]] = (height - bbox[:, [2, 0]])%(height+1)
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(0)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, height)
                target["keypoints"] = keypoints
        return image, target
    
class Resize(object):
    def __init__(self, size):
        self.size = size
        self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.size)])

    def __call__(self, image, target):
        height, width = image.shape[:2]
        image = self.transform(image)
        if "masks" in target:
            target["masks"] = self.transform(target["masks"])
        bboxes = target["boxes"]
        fac_y = self.size/height
        fac_x = self.size/width

        new_bb = []
        for bbox in bboxes: 
            xmin,ymin,xmax,ymax = bbox
            xmin = (xmin*fac_x)#%(self.size+1)
            xmax = (xmax*fac_x)#%(self.size+1)
            ymin = (ymin*fac_y)#%(self.size+1)
            ymax = (ymax*fac_y)#%(self.size+1)
            
            xmin_true = np.min((xmin,xmax))
            xmax_true = np.max((xmin,xmax))
            ymin_true = np.min((ymin,ymax))
            ymax_true = np.max((ymin,ymax))


            """if(xmin<0):
                xmin = self.size+xmin
            if(xmax<0):
                xmax = self.size+xmax
            if(ymin<0):
                ymin = self.size+ymin
            if(ymax<0):
                ymax = self.size+ymax"""
            
            new_bb.append([xmin_true,ymin_true,xmax_true,ymax_true])
            # new_bb.append([ymin,xmin,ymax,xmax])

        if(len(bboxes)== 0):
            new_bb = np.array([]).reshape(-1, 4)

        target["boxes"] = torch.as_tensor(new_bb, dtype=torch.float32)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ToRGB(object):
    def __init__(self):
       pass 

    def __call__(self, image, target):
        # image = image.squeeze()
        image = torch.cat([image,image,image], 0)
        return image, target