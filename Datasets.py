import gzip, pickle
import numpy as np
import glob
import os
import random
from torch.utils.data import Dataset
from Transforms import masking_transform
import pathlib

import sys
sys.path.insert(0,'..')
from Variables import *
from Utils import *

import torch
import wandb
import torchvision


np.random.seed(seed=42)
torch.manual_seed(42)
random.seed(42)

PARAMS = "/params/"


def print_path_stats(paths):
    num_viruses = [int(pathlib.Path(p).stem.split("_")[0]) for p in paths]
    num_viruses = np.array(num_viruses)
    img_without_virus = np.sum(num_viruses == 0)
    # img_without_virus = len([p for p in paths if pathlib.Path(p).stem.startswith("0_")])
    img_with_virus = len(paths) - img_without_virus
    number_bb = np.sum(num_viruses)

    print("Images Total = "+str(len(paths)))
    print("Images with virus = "+str(img_with_virus))
    print("Images without virus = "+str(img_without_virus))
    print("Number of BB = "+str(number_bb))







class AbstractHerpesDataset(Dataset):
    def __init__(self, path, transform, seed, annotation_time, percentage, timings_path, data_paths = [], num_virus = -1, num_imgs = 1, idx = -1, start_idx = 0, preload = True, entities_to_load = ["crops", "labels", "masks", "bboxs"], corruption_probability = 0):
        self.transform = transform
        self.corruption_probability = corruption_probability
        if(self.corruption_probability and (not preload)):
            print("ERROR:: Cannot apply corruption_probability when preload is set to False.")
            import sys 
            sys.exit()
        image_paths = glob.glob(path+"/*") 
        deterministic(seed = seed)
        np.random.shuffle(image_paths) # shuffle images to get different data splits
        paths = []
        for img_path in image_paths:
            files = glob.glob(img_path+"/*.pkl")
            files.sort(key=os.path.getmtime) # get patches by creation time
            paths.extend(files)

        if(len(data_paths)>0):
            print("INFO::Use data_paths")
            self.path = data_paths
            annotation_time = 0
            unique_virus, virus_timings = read_pickle(timings_path)
            for path in data_paths:
                try:
                    num_virus_patch = int(pathlib.Path(path).stem.split("_")[0])
                    time = virus_timings[unique_virus == num_virus_patch]
                    if(np.sum(unique_virus==num_virus_patch)==0):
                        time = num_virus_patch*virus_timings[unique_virus==1]
                    
                    annotation_time += time
                except:
                    print("WARNING::No annotation time is computed. (Should only be done for pseudolabels)")
                    break
            


        else:
            if(annotation_time<0 and percentage<0):
                self.path = paths
            elif(annotation_time>0): 
                unique_virus, virus_timings = read_pickle(timings_path)
                # reduce dataset by annotation time
                self.path = []
                curr_annotation_time = 0
                print("INFO::Pick patches for annotation time of "+str(annotation_time)+"s")
                for path in paths:
                    num_virus_patch = int(pathlib.Path(path).stem.split("_")[0])
                    time = virus_timings[unique_virus == num_virus_patch]
                    curr_annotation_time += time
                    self.path.append(path)
                    if(curr_annotation_time>annotation_time):
                        break
                print("INFO::Picked patches with annotation time: "+str(curr_annotation_time))
            elif(percentage>0):
                self.path = np.random.choice(paths, int(percentage*len(paths)))

        # get only images with 'num_virus' virus particles.
        if(num_virus >= 0):
            str_num_virus = str(num_virus)+"_"
            self.path = [p for p in self.path if pathlib.Path(p).stem.startswith(str_num_virus)] # only get images where one virus is contained
        if(num_virus == -2):
            str_num_virus = "0_"
            self.path = [p for p in self.path if not pathlib.Path(p).stem.startswith(str_num_virus)] # only get images where one virus is contained
            

        if(idx >= 0): # use single image
            self.path = [self.path[idx]]
        elif(num_imgs < 1): # use percentage of images
            num_imgs = int(num_imgs*len(self.path))
            np.random.seed(42)
            r_idx = np.random.randint(0, len(self.path), (int(num_imgs),))
            self.path = (np.array(self.path)[r_idx]).tolist()
            # self.path = self.path[int((num_imgs_path//2)-(num_imgs//2)):int((num_imgs_path//2)+(num_imgs//2)+1)]
        elif(num_imgs>1): # use specified number of images
            np.random.seed(42)
            r_idx = np.random.randint(0, len(self.path), (int(num_imgs),))
            self.path = (np.array(self.path)[r_idx]).tolist()
            print("Use images with IDs: "+str(r_idx))

        if(start_idx):
            self.path = self.path[start_idx-1:]
        
        if(preload):
            try:
                # sets self.crops, self.labels, self.masks, self.bboxes are preloaded
                self.load_from_path(self.path, entities_to_load)
            except:
                pass
        
        # class weights - inspired by Logistic Regression in Rare Events Data, King, Zen, 2001. Similar to sklearn.utils.class_weight.compute_class_weight
        self.class_weights = []    
        n_samples = len(self.path)
        n_classes = 2
        num_no_virus = len([p for p in self.path if pathlib.Path(p).stem.startswith("0_")])
        num_virus = n_samples - num_no_virus
        bin_count = np.array([num_no_virus, num_virus])
        self.class_weights = n_samples / (n_classes * bin_count)
        print("Loaded all data. Number of images: "+str(len(self)))
        print("Class weights: "+str(self.class_weights))
        print("Samples with virus: "+str(num_virus))
        print("Samples without virus: "+str(num_no_virus))


        self.percentage = (len(self.path)/len(paths))*100
        print("INFO::use "+str(self.percentage)+"% of data")
        try:
            wandb.log({"Data/Percentage": self.percentage})
            wandb.log({"Data/Absolute": len(self.path)})
            wandb.log({"Data/AnnotationTime": annotation_time})
            wandb.log({"Data/DataPercentage": percentage})
            wandb.log({"Data/WithVirus": num_virus})
            wandb.log({"Data/NoVirus": num_no_virus})
        except:
            print("WARNING::No wandb logging initialized")
        
        print_path_stats(self.path)





    def load_one(self, idx):
        crop, mask, label, xmin, xmax, ymin, ymax, magnification, pixelsize, p = read_pickle(self.path[idx])
        bbox = [xmin,xmax,ymin,ymax]
        mask = np.array(mask)
        mask = (mask>0.9).astype(np.float32)
        capside_size = compute_capside_size(pixelsize)
        crop = crop.astype(np.float32)
        mask = mask.astype(np.float32)
        return bbox, label, crop, mask, capside_size
        

    def load_from_path(self, paths, entities_to_load = ["crops", "labels", "masks", "bboxs", "virussize"]):
        crops = []
        labels = []
        masks = []
        bboxs = []
        capsidesizes = []
        for i in range(len(paths)): 
            bbox, label, crop, mask, capside_size = self.load_one(i)

            if(self.corruption_probability > 0):
                num_array = np.max((len(bbox[0]), len(label)))
                keep = np.random.choice([True, False], size = (num_array,), p = [1-self.corruption_probability, self.corruption_probability])
                
            if("virussize" in entities_to_load):
                capsidesizes.append(capside_size)
            if("bboxs" in entities_to_load):
                #corrupt labels
                if((self.corruption_probability > 0) and (len(bbox[0])>0)):
                    # bbox = [ b for b, k in zip(bbox, keep) if k ]
                    xmins, ymins, xmaxs, ymaxs = bbox
                    
                    xmins = [ b for b, k in zip(xmins, keep) if k ]
                    ymins = [ b for b, k in zip(ymins, keep) if k ]
                    xmaxs = [ b for b, k in zip(xmaxs, keep) if k ]
                    ymaxs = [ b for b, k in zip(ymaxs, keep) if k ]
                    bbox = [xmins,ymins,xmaxs,ymaxs]

                bboxs.append(bbox)
                # if(len(bbox)!= 4):
                    # print("Length: "+str(len(bbox)))
            if("labels" in entities_to_load):
                # corrupt bboxes
                if((self.corruption_probability > 0) and (len(label)>0)):
                    label = [ l for l, k in zip(label, keep) if k ]
                labels.append(label)
            if("crops" in entities_to_load):
                crops.append(crop)
            mask = np.array(mask)
            if("masks" in entities_to_load):
                masks.append((mask>0.9))
    
        self.crops = crops #np.array(crops).astype(np.float32)
        self.labels = labels # contains strings
        self.masks = masks #np.array(masks).astype(np.float32)
        self.bboxes = bboxs 
        self.capsidesizes = capsidesizes #np.array(capsidesizes).astype(np.int64)

        if(self.corruption_probability>0):
            lengths = [len(l) for l in self.labels]
            reduction = np.sum(lengths)
            print("INFO:: Corruption probability = "+str(self.corruption_probability)+" number of labels have been reduced to "+str(reduction)+" from 2186")
        return 

    def __len__(self):
        return len(self.path)

# Dataset for training the classifier
class Herpes_Classification(AbstractHerpesDataset):
    def __init__(self, path, transform, seed, annotation_time, percentage, timings_path, num_data, preload, data_paths = [], corruption_probability = 0):
        super().__init__(path, transform, seed, annotation_time, percentage, timings_path, data_paths = data_paths, num_imgs = num_data, preload=preload, entities_to_load= ["crops", "labels", "virussize"], corruption_probability = corruption_probability)
        self.preload = preload

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if(self.preload):
            label = self.labels[idx]
            crop = self.crops[idx]
            capside_size = self.capsidesizes[idx]
        else: 
            try:
                _, label, crop, _, _, _, _, _, _, capside_size = self.load_one(idx)
            except:
                _, label, crop, _, capside_size = self.load_one(idx)
        crop = crop.astype(np.float32)
        crop = self.transform(crop)   
        curr_label = np.array(label)
        label = np.array([int(curr_label.shape[0]>0)])
        label = label.astype(np.float32)
        out = {'image': crop, 'label': label, 'capsidsize':capside_size}
        return out

# Dataset for Pseudolabel generation
class HerpesLabelGeneration_Dataset(AbstractHerpesDataset):
    def __init__(self, path, transform, seed, annotation_time, percentage, timings_path, preload, corrupt_size =-1,  data_paths = [], num_virus = -1, num_imgs = 1, idx = -1, start_idx=0, entities_to_load = ["crops", "labels", "masks", "bboxs", "virussize"]):
        super().__init__(path, transform, seed, annotation_time, percentage, timings_path, data_paths = data_paths, num_imgs=num_imgs, idx=idx, num_virus=num_virus, start_idx=start_idx, preload=preload, entities_to_load=entities_to_load)
        self.preload = preload
        self.corrupt_size = corrupt_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            # get image
            crop = self.crops[idx]
            label_img = self.labels[idx]
            mask = self.masks[idx]
            bbox = self.bboxes[idx]
            capsidesize = self.capsidesizes[idx]
            
        else: 
            bbox, label_img, crop, mask, capsidesize = self.load_one(idx)
        crop = self.transform(crop)

        # get GT mask
        gt_mask = torch.from_numpy(mask).float()

        # get label
        label = np.zeros((3,))
        curr_label = np.array(label_img)
        if(curr_label.shape[0]>0):
            label[0] = np.sum(curr_label == NAMES[0])
            label[1] = np.sum(curr_label == NAMES[1])
            label[2] = np.sum(curr_label == NAMES[2])
        label = np.sum(label)
        label = torch.tensor(label).float()

        # get bounding boxes + locations
        xmins,xmaxs,ymins,ymaxs = bbox
        boxes = []
        locations = [] 
        for i,(xmin, xmax, ymin, ymax) in enumerate(zip(xmins, xmaxs, ymins, ymaxs)):
            boxes.append([xmin, ymin, xmax, ymax])

            x = xmin + ((xmax-xmin)/2)
            y = ymin + ((ymax-ymin)/2)
            locations.append([x,y])


        num_objs = len(xmins)
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            box_labels = torch.tensor([], dtype=torch.int64)
            area = torch.tensor(0)

            locations = np.array([]).reshape(-1, 2)
            locations = torch.as_tensor(locations, dtype=torch.float32)
        else: 
            box_labels = torch.ones((np.max((num_objs,1)),), dtype=torch.int64)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            locations = torch.as_tensor(locations, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        if(self.corrupt_size>0):
            capsidesize = (1+self.corrupt_size)*capsidesize
        
        out = {}
        out['image'] = crop
        out['gt_mask'] = gt_mask
        out['label'] = label
        out['path'] = self.path[idx]
        out['capsideradius'] = int(round(capsidesize/2))
        out['locations'] = locations
     
        target = {}
        target["boxes"] = boxes
        target["labels"] = box_labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return out, target


# FRCNN Datasets
class HerpesBBDataset_GT(AbstractHerpesDataset):
    def __init__(self, path, transform, seed, annotation_time, percentage, timings_path, preload, data_paths = [], loc=False, num_virus = -1, num_imgs = 1, entities_to_load = ["crops", "bboxs", "virussize"], corruption_probability = 0):
        super().__init__(path, transform, seed, annotation_time, percentage, timings_path, data_paths = data_paths, num_virus = num_virus, num_imgs = num_imgs, preload=preload, entities_to_load= entities_to_load, corruption_probability = corruption_probability)
        self.preload = preload
        self.loc = loc
        print("Loaded all data. Number of images: "+str(len(self)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            crop = self.crops[idx].astype(np.float32)
            xmins,xmaxs,ymins,ymaxs = self.bboxes[idx]
            capside_size = self.capsidesizes[idx]
        else:
            try:
                bbox, _, crop, _, capside_size = self.load_one(idx)
            except: 
                bbox, _, crop, _, capside_size, _, _, _, _, _ = self.load_one(idx)
            xmins,xmaxs,ymins,ymaxs = bbox
        
        crop = crop[None,:,:]

        # bboxes
        num_objs = len(xmins)
        boxes = []
        radius = capside_size/2
        for i,(xmin, xmax, ymin, ymax) in enumerate(zip(xmins, xmaxs, ymins, ymaxs)):
            # use center as loc 
            if(self.loc):
                center_x = 0.5*(xmax - xmin) + xmin
                center_y = 0.5*(ymax - ymin) + ymin
                xmin = np.max((int(center_x - radius), 0))
                xmax = np.min((int(center_x + radius), IMG_SIZE[0]))    
                ymin = np.max((int(center_y - radius), 0))
                ymax = np.min((int(center_y + radius), IMG_SIZE[1]))  

            boxes.append([xmin, ymin, xmax, ymax])
        
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            labels = torch.tensor([], dtype=torch.float32)
            area = torch.tensor(0)
        else: 
            labels = torch.ones((np.max((num_objs,1)),), dtype=torch.float32) # as probabilites

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # convert everything into a torch.Tensor
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['path'] = self.path[idx]
        
        crop, target = self.transform(crop.transpose(1,2,0),target)
        
        return crop, target

class HerpesBBDataset_Ours(AbstractHerpesDataset):

    def __init__(self, path, transform, seed, annotation_time, percentage, timings_path, preload, data_paths = [], threshold = -1, num_virus = -1, num_imgs = 1):
        super().__init__(path, transform, seed, annotation_time, percentage, timings_path, data_paths = data_paths, num_imgs = num_imgs, preload=preload, num_virus=num_virus)
        self.preload = preload
        self.threshold = threshold
        if(self.preload):
            self.load_from_path()
        print("Loaded all data. Number of images: "+str(len(self)))
    
    def load_one(self, idx):
        img_size, positions, bb_scores, capside_radius, prediction, time_delta, crop_path, iou_value, input_img, target_boxes, target_labels, model_path = read_pickle(self.path[idx])
        crop_path = crop_path[2:-2]
        crop, _, _, _, _, _, _, _, _, _ = read_pickle(crop_path)
        crop = crop.astype(np.float32)

        # convert positions to BB
        bbox = []
        if(not np.any(positions==-1)):
            for score, position in zip(bb_scores, positions): 
                if((score >= self.threshold) or (self.threshold == -1)):
                    xmin = np.max((position[0]-capside_radius, 0))
                    xmax = np.min((position[0]+capside_radius, img_size[0]))
                    ymin = np.max((position[1]-capside_radius, 0))
                    ymax = np.min((position[1]+capside_radius, img_size[1]))
                    bbox.append([int(xmin),int(ymin),int(xmax),int(ymax)])
        return crop, bbox, bb_scores

        
    def load_from_path(self):
        crops = []
        scores = []
        bboxes = []
        for idx in range(len(self.path)): 
            crop, bbox, bb_scores = self.load_one(idx)
            crops.append(crop)
            scores.append(bb_scores)
            bboxes.append(bbox)
        length = len(crops)
        self.crops = crops
        self.bboxes = bboxes 
        self.scores = scores
        self.length = length
        return 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            crop = self.crops[idx].astype(np.float32)
            boxes = self.bboxes[idx]
            bb_scores = self.scores[idx]
        else:
            crop, bbox, bb_scores = self.load_one(idx)
            boxes = bbox
        
        crop = crop[None,:,:]
        # crop = self.transform(crop.transpose(1,2,0))  

        num_objs = len(boxes)
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            area = torch.tensor(0)
        else:
            if(self.threshold == -1): 
                # as probabilities
                labels = torch.tensor(bb_scores)
            else: 
                labels = torch.ones((np.max((num_objs,1)),), dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # convert everything into a torch.Tensor
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        crop, target = self.transform(crop.transpose(1,2,0),target)

        return crop, target
    
class Pseudolabels(Dataset):
    def __init__(self, seed, path, transform, filter, size_range,  threshold = -1, preload = True):
        self.percentage = -1
        self.path = path
        self.threshold = threshold
        self.transform = transform 
        self.filter = filter
        self.preload = preload
        self.size_range = size_range

        image_paths = glob.glob(path+"/*") 
        deterministic(seed = seed)
        np.random.shuffle(image_paths) # shuffle images to get different data splits
        paths = []
        for img_path in image_paths:
            files = glob.glob(img_path+"/*.pkl")
            files.sort(key=os.path.getmtime) # get patches by creation time
            paths.extend(files)    

        self.path = paths           
        
        if(preload):
            # sets self.crops, self.labels, self.masks, self.bboxes are preloaded
            self.load_from_path()
        
        # class weights - inspired by Logistic Regression in Rare Events Data, King, Zen, 2001. Similar to sklearn.utils.class_weight.compute_class_weight
        self.class_weights = []    
        n_samples = len(self.path)
        n_classes = 2
        num_no_virus = len([p for p in self.path if pathlib.Path(p).stem.startswith("0_")])
        num_virus = n_samples - num_no_virus
        bin_count = np.array([num_no_virus, num_virus])
        self.class_weights = n_samples / (n_classes * bin_count)
        print("Loaded all data. Number of images: "+str(len(self)))
        print("Class weights: "+str(self.class_weights))
        print("Samples with virus: "+str(num_virus))
        print("Samples without virus: "+str(num_no_virus))

        self.percentage = (len(self.path)/len(paths))*100
        print("INFO::use "+str(self.percentage)+"% of data")
        try:
            wandb.log({"Data/Percentage": self.percentage})
            wandb.log({"Data/Absolute": len(self.path)})
            wandb.log({"Data/AnnotationTime": -1})
            wandb.log({"Data/DataPercentage": self.percentage})
            wandb.log({"Data/WithVirus": num_virus})
            wandb.log({"Data/NoVirus": num_no_virus})
        except:
            print("WARNING::No wandb logging initialized")

    def load_one(self, idx):
        crop,capside_size,gt_boxes,predicted_boxes,probabilities = read_pickle(self.path[idx])
        crop = crop.astype(np.float32)

        if(len(gt_boxes)==0):
            predicted_boxes = []
            probabilities = None

        filtered_boxes = []
        for box in predicted_boxes:
            xmin,ymin,xmax,ymax = box

            xmin = np.max((0,xmin))
            xmax = np.min((xmax,IMG_SIZE[0]))
            ymin = np.max((0,ymin))
            ymax = np.min((ymax,IMG_SIZE[0]))


            w = int(xmax)-int(xmin)
            h = int(ymax)-int(ymin)

            max_w = (1+self.size_range)*capside_size
            min_w = (1-self.size_range)*capside_size

            if((w > 0) and (h > 0)):
                if(not self.filter):
                    filtered_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)]) 

                else:
                    # don't add bb if size does not match virus size
                    if(((w>=min_w) and (w<=max_w)) or ((h>=min_w) and (h<=max_w))):
                        filtered_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])  

        predicted_boxes = filtered_boxes        
        return crop, gt_boxes, predicted_boxes, probabilities        
    
    def load_from_path(self):
        crops = []
        scores = []
        bboxes = []
        for idx in range(len(self.path)): 
            crop, gt_bbox, bbox, bb_scores = self.load_one(idx)
            crops.append(crop)
            scores.append(bb_scores)
            bboxes.append(bbox)
        length = len(crops)
        self.crops = crops
        self.bboxes = bboxes 
        self.scores = scores
        self.length = length
        return
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            crop = self.crops[idx].astype(np.float32)
            boxes = self.bboxes[idx]
        else:
            crop, gt_box, bbox, bb_scores = self.load_one(idx)
            boxes = bbox
        
        if(len(crop.shape) == 3 and (crop.shape[2] == 3)):
            crop = crop[:,:,0][None,:,:]
        else: 
            crop = crop[None,:,:]
        
        # crop = self.transform(crop.transpose(1,2,0))  

        num_objs = len(boxes)
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            area = torch.tensor(0)
        else:
            if(self.threshold == -1): 
                # as probabilities
                labels = torch.tensor(bb_scores)
            else: 
                labels = torch.ones((np.max((num_objs,1)),), dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # convert everything into a torch.Tensor
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        
        crop, target = self.transform(crop.transpose(1,2,0),target)
        fixed_boxes = []
        for box in target['boxes']:
            xmin,ymin,xmax,ymax = box
   
            w = xmax - xmin
            h = ymax - ymin
            if((w <= 0) or (h <= 0)):
                continue
            else: 
                fixed_boxes.append([xmin, ymin, xmax, ymax])
        if(len(fixed_boxes) == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
        else: 

            boxes = torch.as_tensor(fixed_boxes, dtype=torch.float32)
        target['boxes'] = boxes
        # print(target)
        return crop, target

    


class AbstractTEMDataset(Dataset):
    def __init__(self, path, transform, seed, annotation_time, task, data_paths = [], num_virus = -1, num_imgs = 1, idx = -1, start_idx = 0, preload = True, entities_to_load = ["crops", "labels", "masks", "bboxs"]):
        self.transform = transform
        self.task = task # one of bin, loc
        image_paths = glob.glob(path+"/*") 
        
        deterministic(seed = seed)
        np.random.shuffle(image_paths) # shuffle images to get different data splits
        paths = []
        for img_path in image_paths:
            files = glob.glob(img_path+"/*.pkl")
            files.sort(key=os.path.getmtime) # get patches by creation time
            paths.extend(files)

        if(len(data_paths)>0):
            print("INFO::Use data_paths")
            self.path = data_paths
            annotation_time = -1   


        else:
            if(annotation_time<0):
                self.path = paths
                # if annotation_time = -2 and data split is for training, add synthetic data 
                if((annotation_time == -2) and (pathlib.Path(path).parent.stem == 'train')):
                    synthetic_paths = pathlib.Path(path).parent.parent / "synthetic"
                    synthetic_paths = glob.glob(str(synthetic_paths)+"/*.pkl")
                    self.path.extend(synthetic_paths)
            else: 
                # reduce dataset by annotation 
                print("INFO::Pick patches for annotation time of "+str(annotation_time)+"s")

                img_n_viruses = [int(pathlib.Path(p).stem.split("_")[0]) for p in paths]
                img_n_viruses_unique = np.unique(img_n_viruses)
                occurences = np.array([np.sum(img_n_viruses==unique) for unique in img_n_viruses_unique])
                probabilities = occurences/np.sum(occurences)

                self.path = []
                combined_annotation_time = 0
                for unique, occurence, probability in zip(img_n_viruses_unique, occurences, probabilities):
                    str_num_virus = str(unique)
                    curr_paths = [p for p in paths if pathlib.Path(p).stem.startswith(str_num_virus)]
                    _, _, _, _, t_loc, t_classification, _, _ = read_pickle(curr_paths[0])
                    if(self.task == "loc"):
                        t = t_loc
                    elif(self.task == "bin"):
                        t = t_classification

                        


                    time_to_annotate = annotation_time*probability#occurence*t*probability
                    curr_annotation_time = 0
                    j = 0
                    while(curr_annotation_time<time_to_annotate):
                        self.path.append(curr_paths[j])
                        curr_annotation_time += t
                        j += 1
                    combined_annotation_time += curr_annotation_time
                print("INFO::Picked patches with annotation time: "+str(combined_annotation_time))

        # get only images with 'num_virus' virus particles.
        if(num_virus >= 0):
            str_num_virus = str(num_virus)+"_"
            self.path = [p for p in self.path if pathlib.Path(p).stem.startswith(str_num_virus)] # only get images where one virus is contained
        if(num_virus == -2):
            str_num_virus = "0_"
            self.path = [p for p in self.path if not pathlib.Path(p).stem.startswith(str_num_virus)] # only get images where one virus is contained
            
        if(idx >= 0): # use single image
            self.path = [self.path[idx]]
        elif(num_imgs < 1): # use percentage of images
            num_imgs = int(num_imgs*len(self.path))
            np.random.seed(42)
            r_idx = np.random.randint(0, len(self.path), (int(num_imgs),))
            self.path = (np.array(self.path)[r_idx]).tolist()
            # self.path = self.path[int((num_imgs_path//2)-(num_imgs//2)):int((num_imgs_path//2)+(num_imgs//2)+1)]
        elif(num_imgs>1): # use specified number of images
            np.random.seed(42)
            r_idx = np.random.randint(0, len(self.path), (int(num_imgs),))
            self.path = (np.array(self.path)[r_idx]).tolist()
            print("Use images with IDs: "+str(r_idx))

        if(start_idx):
            self.path = self.path[start_idx-1:]
        
        if(preload):
            # self.crops, self.labels, self.bboxes are preloaded
            self.load_from_path(self.path, entities_to_load)
        
        # class weights - inspired by Logistic Regression in Rare Events Data, King, Zen, 2001. Similar to sklearn.utils.class_weight.compute_class_weight
        self.class_weights = []    
        n_samples = len(self.path)
        n_classes = 2
        num_no_virus = len([p for p in self.path if pathlib.Path(p).stem.startswith("0_")])
        num_virus = n_samples - num_no_virus
        bin_count = np.array([num_no_virus, num_virus])
        self.class_weights = n_samples / (n_classes * bin_count)
        print("Loaded all data. Number of images: "+str(len(self)))
        print("Class weights: "+str(self.class_weights))
        print("Samples with virus: "+str(num_virus))
        print("Samples without virus: "+str(num_no_virus))

        self.percentage = (len(self.path)/len(paths))*100
        print("INFO::use "+str(self.percentage)+"% of data")
        try:
            wandb.log({"Data/Percentage": self.percentage})
            wandb.log({"Data/Absolute": len(self.path)})
            wandb.log({"Data/AnnotationTime": annotation_time})
            wandb.log({"Data/WithVirus": num_virus})
            wandb.log({"Data/NoVirus": num_no_virus})
        except:
            print("WARNING::No wandb logging initialized")
        print_path_stats(self.path)
        



    def load_one(self, idx):
        crop, locations, bbox, virus_radius, t_loc, t_classification, pixelsize, p = read_pickle(self.path[idx])
        crop = crop.astype(np.float32)
        label = 0
        if(len(bbox)>0):
            label = 1
        return bbox, label, crop, t_loc, t_classification, virus_radius
        

    def load_from_path(self, paths, entities_to_load = ["crops", "labels", "bboxs", "capsidesizes"]):
        crops = []
        labels = []
        bboxs = []
        capsidesizes = []
        for i in range(len(paths)): 
            bbox, label, crop, _, _, virus_radius = self.load_one(i)
            if("capsidesizes" in entities_to_load):
                capsidesizes.append(virus_radius*2)
            if("bboxs" in entities_to_load):
                bboxs.append(bbox)
                if(len(bbox)!= 4):
                    print("Length: "+str(len(bbox)))
            if("labels" in entities_to_load):
                labels.append(label)
            if("crops" in entities_to_load):
                crops.append(crop)

        self.crops = np.array(crops).astype(np.float32)
        self.labels = labels # contains strings
        self.bboxes = bboxs 
        self.capsidesizes = np.array(capsidesizes).astype(np.int8)
        return 

    def __len__(self):
        return len(self.path)
    
def positions_from_BBs(bboxes): 
    positions = []
    for box in bboxes:
        xmin,ymin,xmax,ymax = box 
        x = ((xmax-xmin)/2)+xmin
        y = ((ymax-ymin)/2)+ymin
        positions.append([x,y])
    return positions
    
#from Transforms import norm_resnet101
class TEM_Classification(AbstractTEMDataset):
    def __init__(self, path, transform, seed, annotation_time, num_data, preload, data_paths = []):
        super().__init__(path, transform, seed, annotation_time, "bin", data_paths = data_paths, num_imgs = num_data, preload=preload, entities_to_load= ["crops", "labels"])
        self.preload = preload

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if(self.preload):
            label = self.labels[idx]
            crop = self.crops[idx]
        else: 
            bboxes, label, crop, t_loc, t_classification, virus_radius = self.load_one(idx)
            capside_size = 2*virus_radius
        
        crop = crop.astype(np.float32)
        gt_mask = torch.from_numpy(np.zeros_like(crop))
        label = np.array([label])
        label = label.astype(np.float32)

        crop = self.transform(crop)

        if(len(bboxes)>0):
            gt_mask = generate_masks_from_boxes(bboxes)
            gt_mask = torch.sum(gt_mask,dim=0).squeeze()
        
        positions = positions_from_BBs(bboxes)
        positions = torch.as_tensor(positions,  dtype=torch.float32).reshape(-1,2)

        padded_pos = torch.zeros((50,2)) -1
        padded_pos[:positions.shape[0],:] = positions
 
            
        out = {'image': crop, 'label': label, 'capsidsize':capside_size, 'gt_mask': gt_mask, 'loc': padded_pos}
        return out
    
class TEMLabelGeneration_Dataset(AbstractTEMDataset):
    def __init__(self, path, transform, seed, annotation_time, preload, data_paths = [], num_virus = -1, num_imgs = 1, idx = -1, start_idx=0, entities_to_load = ["crops", "labels", "masks", "bboxs", "virussize"]):
        super().__init__(path, transform, seed, annotation_time, "bin", data_paths = data_paths, num_imgs=num_imgs, idx=idx, num_virus=num_virus, start_idx=start_idx, preload=preload, entities_to_load=entities_to_load)
        self.preload = preload

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            # get image
            crop = self.crops[idx]
            label_img = self.labels[idx]
            boxes = self.bboxes[idx]
            capsidesize = self.capsidesizes[idx]
        else: 
            boxes, label_img, crop, t_loc, t_classification, virus_radius = self.load_one(idx)
            capsidesize = virus_radius*2
        crop = self.transform(crop)

        # get label
        label = np.array([label_img])
        label = label.astype(np.float32)
        label = torch.tensor(label).float()

        # get bounding boxes
        locations = []
        for (xmin, ymin, xmax, ymax) in boxes:

            x = xmin + ((xmax-xmin)/2)
            y = ymin + ((ymax-ymin)/2)
            locations.append([x,y])


        num_objs = len(boxes)
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            locations = np.array([]).reshape(-1, 2)
            locations = torch.as_tensor(locations, dtype=torch.float32)
            box_labels = torch.tensor([], dtype=torch.int64)
            area = torch.tensor(0)
        else: 
            box_labels = torch.ones((np.max((num_objs,1)),), dtype=torch.int64)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            locations = torch.as_tensor(locations, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)
        
        # get GT mask
        mask = np.zeros(IMG_SIZE)
        for b in boxes: 
            xmin,ymin,xmax,ymax = b
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        gt_mask = torch.from_numpy(mask).float()


        out = {}
        out['image'] = crop
        out['gt_mask'] = gt_mask
        out['label'] = label
        out['path'] = self.path[idx]
        out['capsideradius'] = int(round(capsidesize/2))
        out['locations'] = locations

     
        target = {}
        target["boxes"] = boxes
        target["labels"] = box_labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return out, target
    

# FRCNN Datasets
class TEMBBDataset_GT(AbstractTEMDataset):
    def __init__(self, path, transform, seed, annotation_time, preload, data_paths = [], num_virus = -1, num_imgs = 1, entities_to_load = ["crops", "bboxs"]):
        super().__init__(path, transform, seed, annotation_time, "loc", data_paths = data_paths, num_virus = num_virus, num_imgs = num_imgs, preload=preload, entities_to_load= entities_to_load)
        self.preload = preload
        print("Loaded all data. Number of images: "+str(len(self)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            crop = self.crops[idx].astype(np.float32)
            boxes = self.bboxes[idx]
        else:
            boxes, label_img, crop, t_loc, t_classification, virus_radius = self.load_one(idx)

        crop = crop[None,:,:]
       
        # bboxes
        num_objs = len(boxes)
        
        
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            labels = torch.tensor([], dtype=torch.float32)
            area = torch.tensor(0)
        else: 
            labels = torch.ones((np.max((num_objs,1)),), dtype=torch.float32) # as probabilites

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # convert everything into a torch.Tensor
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        crop, target = self.transform(crop.transpose(1,2,0),target)
        
        return crop, target
    

class TEMBBDataset_Ours(AbstractTEMDataset):
    def __init__(self, path, transform, seed, annotation_time, preload, data_paths = [], threshold = -1, num_virus = -1, num_imgs = 1):
        super().__init__(path, transform, seed, annotation_time, "bin", data_paths = data_paths, num_imgs = num_imgs, preload=preload, num_virus=num_virus)
        self.preload = preload
        self.threshold = threshold
        print("Loaded all data. Number of images: "+str(len(self)))
    
    def load_one(self, idx):
        img_size, positions, bb_scores, capside_radius, prediction, time_delta, crop_path, iou_value, input_img, target_boxes, target_labels, model_path = read_pickle(self.path[idx])
        crop_path = crop_path[2:-2]

        crop, _, _, _, _, _, _, _ = read_pickle(crop_path)
        crop = crop.astype(np.float32)

        # convert positions to BB
        bbox = []
        if(not np.any(positions==-1)):
            for score, position in zip(bb_scores, positions): 
                if((score >= self.threshold) or (self.threshold == -1)):
                    xmin = np.max((position[0]-capside_radius, 0))
                    xmax = np.min((position[0]+capside_radius, img_size[0]))
                    ymin = np.max((position[1]-capside_radius, 0))
                    ymax = np.min((position[1]+capside_radius, img_size[1]))
                    bbox.append([int(xmin),int(ymin),int(xmax),int(ymax)])
        return crop, bbox, bb_scores

        
    def load_from_path(self):
        crops = []
        scores = []
        bboxes = []
        for idx in range(len(self.path)): 
            crop, bbox, bb_scores = self.load_one(idx)
            crops.append(crop)
            scores.append(bb_scores)
            bboxes.append(bbox)
        length = len(crops)
        self.crops = crops
        self.bboxes = bboxes 
        self.scores = scores
        self.length = length
        return 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if(self.preload):
            crop = self.crops[idx].astype(np.float32)
            boxes = self.bboxes[idx]
        else:
            crop, bbox, bb_scores = self.load_one(idx)
            boxes = bbox
        
        crop = crop[None,:,:]
        # crop = self.transform(crop.transpose(1,2,0))  

        num_objs = len(boxes)
        if(num_objs == 0):
            boxes = np.array([]).reshape(-1, 4)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            area = torch.tensor(0)
        else:
            if(self.threshold == -1): 
                # as probabilities
                labels = torch.tensor(bb_scores)
            else: 
                labels = torch.ones((np.max((num_objs,1)),), dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # convert everything into a torch.Tensor
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((np.max((num_objs,1)),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        crop, target = self.transform(crop.transpose(1,2,0),target)

        return crop, target