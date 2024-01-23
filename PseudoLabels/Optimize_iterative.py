import argparse
import wandb
import torch
import numpy as np
from copy import deepcopy
import torch
from scipy import ndimage
import os
from tqdm import tqdm

import sys
sys.path.insert(0,'..')
sys.path.insert(0, '../Detector/')
from Utils import *
from Utils_Eval import *
from Variables import *
from .Models import TranslationMatrix_iterative
from .Optimize import Optimizer, loss_fct
from GradCAM import GradCAM
import selective_search

import matplotlib.patches as patches



SAVE_GRADIENT_IMAGES = False

SHOW_IMG_GRADIENTS = False
NUM_GRADIENTS = 24

class OptimizerIter(Optimizer):
    def __init__(self, args, log_path, data_split, model, gradcam_model, data_paths, seed):
        title="Iterative_"
        super().__init__(args, log_path, data_split, title, model, gradcam_model, data_paths, seed)
        if(args.dataset == "herpes"):
            self.max_num_obj = MAX_NUM_OBJ_HERPES
        else: 
            self.max_num_obj = 20 

    
    def init_pos_cam(self, cam):  
        if(cam.max() == cam.min()): 
            r1 = -1
            r2 = 1
            x,y = (r1 - r2) * torch.rand((2,)) + r2
        else: 
            k = int((IMG_SIZE[0]*IMG_SIZE[1])*0.01)
            ind = np.argpartition(cam.reshape(-1), -k)[-k:]
            t = np.min(cam.reshape(-1)[ind])
            cam[cam<t] = 0
            com = ndimage.center_of_mass(cam.squeeze())
            x = com[1]
            y = com[0]

            if(cam[0,int(y),int(x)]==0):
                _, y,x = np.unravel_index(cam.argmax(), cam.shape)
            
            plt.close()
            plt.clf()
            plt.figure()
            plt.imshow(cam.squeeze())
            plt.scatter(x,y)
            wandb.log({"Init/img ": wandb.Image(plt)})
            plt.close()            

            x = (2*(x/IMG_SIZE[0]))-1
            y = (2*(y/IMG_SIZE[0]))-1

        
        return torch.FloatTensor([x, y])  

    def filter_boxes(self, boxes, capside_radius):
        new_boxes = []
        for box in boxes: 
            xmin, ymin, xmax, ymax = box
            x_size = xmax-xmin
            y_size = ymax-ymin
            if(((x_size > 0.8*2*capside_radius) and (x_size < 1.2*2*capside_radius)) and ((y_size > 0.8*2*capside_radius) and (y_size < 1.2*2*capside_radius))):
                new_boxes.append([xmin,ymin,xmax,ymax])
        return new_boxes
    
    def remove_detected_locations(self,result_pos, batch):
        curr_mask = generate_masks_from_positions(result_pos[None,:], int(batch['capsideradius'].numpy()))
        batch['gt_mask'] = np.clip(batch['gt_mask'].squeeze()-curr_mask.squeeze(),0,1)
        new_locations = []
        # print(f"Locations before: {batch['locations']}")
        for loc in batch['locations'][0]:
            lx,ly = loc 
            x,y = result_pos 
            distance = np.sqrt((x-lx)**2 + (y-ly)**2)
            # print(f"Distance {distance} with radius {batch['capsideradius']}")
            if(distance < batch['capsideradius']):
                new_locations.append([-1,-1])

            else: 
                new_locations.append([lx,ly])
        
        batch['locations'][0] = torch.as_tensor(new_locations, dtype=torch.float32).reshape(-1,2)
        # print(f"Locations after: {batch['locations']}")
        return batch
    
    def init_pos_selectivesearch(self, curr_img, capsid_radius): 
        boxes = selective_search.selective_search(curr_img.squeeze().permute(1,2,0).numpy(), mode=self.args.selective_search_mode)
        if(self.args.selective_search_topn>0):
            boxes = selective_search.box_filter(boxes, min_size=0.7*2*capsid_radius, topN=self.args.selective_search_topn)

        boxes = self.filter_boxes(boxes, capsid_radius) 
        if(len(boxes)>0):
            masks = generate_masks_from_boxes(boxes)
            scores = self.get_scores_batchified(masks, curr_img, self.model, (capsid_radius*2).to(DEVICE))
            if(np.max(scores)< 0.01):
                return torch.Tensor([False]), len(scores)
            max_idx = np.argmax(scores)
            xmin,ymin,xmax,ymax = boxes[max_idx]
            x = xmin+((xmax-xmin)/2)
            y = ymin+((ymax-ymin)/2)

            plt.close()
            plt.clf()
            fig, ax = plt.subplots()
            ax.imshow(curr_img.squeeze().permute(1,2,0))
            ax.scatter(x,y)
            rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (xmax-xmin), linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            wandb.log({"Init/img ": wandb.Image(plt)})
            plt.close() 

            x = (2*(x/IMG_SIZE[0]))-1
            y = (2*(y/IMG_SIZE[0]))-1
            return torch.FloatTensor([x, y]), len(scores)


        else: 
            return torch.Tensor([False]), 0
        
     
    def image_gradients(self, batch, curr_img, iters, scaler, i):
        pos_x = np.linspace(-1,1, NUM_GRADIENTS)
        pos_y = np.linspace(-1,1, NUM_GRADIENTS)
        x,y = np.meshgrid(pos_x, pos_y)
        positions = np.stack([x,y], axis=-1).reshape(-1,2)

        img_positions_x = []
        img_positions_y = []

        img_gradients_x = []
        img_gradients_y = []

        self.std_fac, self.start_std, self.smallest_max_gradient = self.get_std(batch['capsideradius'])

        e = 1
        es = [10, 20, 30, 40, 50] #, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        # es = [25, 50]
        avg_gradients_magnitude = []
        std_gradients_magnitude = []
        for e in es:
            img_positions_x = []
            img_positions_y = []

            img_gradients_x = []
            img_gradients_y = []

            magnitude = []
            for pos in tqdm(positions):
                pos = torch.from_numpy(pos)
                translation = TranslationMatrix_iterative(init_pos=pos, gaussian_pdf = self.args.pseudolabels_gaussian_pdf).to(DEVICE)
                img_pos = translation.get_pixel_translation()
                img_positions_x.append(img_pos[0])
                img_positions_y.append(img_pos[1])

                optim_t, scheduler_t = self.init_optimizer(translation, self.args.lr_t)

                optim_t.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.args.pseudolabels_use_amp): 
                    std = self.start_std*torch.exp(torch.tensor(-1*self.std_fac*e))  
                    std.requires_grad = False
                    transformed_mask = translation(std.to(DEVICE), batch['capsideradius'].to(DEVICE)) 
                    loss = loss_fct(transformed_mask, curr_img, batch['gt_mask'], self.model, (batch['capsideradius']*2).to(DEVICE), self.act_fct, self.bg, self.args, self.norm_transform)     
                scaler.scale(loss).backward()
                scaler.step(optim_t)
                old_scaler = scaler.get_scale()
                scaler.update()
                if((scheduler_t != None) and (self.args.scheduler != "plateau") and (scaler.get_scale() == old_scaler)):
                    scheduler_t.step()
                
                gradients = translation.translation.grad.cpu().clone().numpy()
                maximum_gaussian_gradient = 1
                grad_x = -1*gradients[0]/maximum_gaussian_gradient
                grad_y = -1*gradients[1]/maximum_gaussian_gradient
                img_gradients_x.append(grad_x)
                img_gradients_y.append(grad_y)
                magnitude.append(np.sqrt(grad_x**2 + grad_x**2))

                
            img_gradients_x =  [0 if x != x else x for x in img_gradients_x]
            img_gradients_y =  [0 if x != x else x for x in img_gradients_y]
            img_positions_x =  [0 if x != x else x for x in img_positions_x]
            img_positions_y =  [0 if x != x else x for x in img_positions_y]

            plt.figure()
            plt.imshow(min_max(curr_img.squeeze().permute(1,2,0)))
            plt.scatter(img_positions_x, img_positions_y, color="red", s = 5)
            if(self.args.loss == "logit"):
                s = 200
            elif(self.args.loss == "score"): 
                s = 100
            else: 
                s = 30
            plt.quiver(img_positions_x, img_positions_y, img_gradients_x, img_gradients_y, color="red", scale = s) # scale = 200) for logits #scale = 30) for scores 
            plt.xlim(0,IMG_SIZE[0])
            plt.ylim(0,IMG_SIZE[1]) 
            plt.axis('off')

            plt.savefig("./DebugImages/GradientFields/Img_"+str(i)+"_Iter_"+str(e)+".jpg", dpi = 300)
            avg_gradients_magnitude.append(np.nanmax(magnitude))
            std_gradients_magnitude.append(np.nanstd(magnitude))

        return


    def train_translation(self, batch, i):
        
        curr_img = batch['image']
        remaining_virus_score = 1
        positions = []
        iters = 0
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.pseudolabels_use_amp) # for 16 bit precision

        training_epochs = 0

        if(SHOW_IMG_GRADIENTS):
            self.image_gradients(batch, curr_img, iters, scaler, i)
            exit()
        
        # continue masking as long as there is still one virus detected
        while((remaining_virus_score>self.args.pseudolabel_threshold) and (iters<self.max_num_obj)):
            if(SAVE_GRADIENT_IMAGES):
                os.makedirs("./DebugImages/Gradients_"+str(iters)+"/", exist_ok=True)
            print("Iters: "+str(iters))
            self.std_fac, self.start_std, self.smallest_max_gradient = self.get_std(batch['capsideradius'])
            translation, epoch = self.train_iter(batch, curr_img, iters, scaler, iters)
            if(translation != False):
                result_pos = translation.get_pixel_translation()
                batch = self.remove_detected_locations(result_pos, batch)


            training_epochs += (epoch + 1)
            if(translation == False): 
                # attention map could not find virus -> hence stop.
                positions = np.array(positions)
                num_virus = positions.shape[0]
                if(num_virus == 0):
                    positions = np.array([-1,-1]).astype(np.int16)
                    scores = np.array([])
                    num_virus = 0
                else:  
                    positions, scores, num_virus = self.select_masks_by_iou(positions, batch['capsideradius']*2, batch['image'], self.args.nms_max_iou, individual_masks = True, gt_masks = batch['gt_mask'])
                    scores = self.recompute_scores(positions, batch['capsideradius'], batch['image'], batch['gt_mask'])
                
                if(SAVE_GRADIENT_IMAGES):
                    exit()
                
                return positions, scores, num_virus, training_epochs
            
            

            # model_in = mask_input(transformed_mask, curr_img.squeeze(), self.args.masking, self.bg, self.norm_transform)
            # detected_virus_score = self.act_fct(self.model(model_in.to(DEVICE)))
            detected_virus_score = self.recompute_scores([translation.get_pixel_translation()], batch['capsideradius'], batch['image'], batch['gt_mask'])[0]

            if(detected_virus_score < self.args.pseudolabel_threshold):
                # mask does not contain more virus -> hence stop.
                positions.append(translation.get_pixel_translation()) 
                positions = np.array(positions)
                num_virus = positions.shape[0]
                positions, scores, num_virus = self.select_masks_by_iou(positions, batch['capsideradius']*2, batch['image'], self.args.nms_max_iou, individual_masks = True, gt_masks = batch['gt_mask'])
                scores = self.recompute_scores(positions, batch['capsideradius'], batch['image'], batch['gt_mask'])
                if(SAVE_GRADIENT_IMAGES):
                    exit()
                return positions, scores, num_virus, training_epochs
            else:          
                transformed_mask = generate_masks_from_positions([translation.get_pixel_translation()], int(batch['capsideradius'].numpy()))
                transformed_mask = transformed_mask.int()             
                curr_img = self.stds*curr_img.cpu()+self.means # invert normalization for further computations
                # pos = translation.get_pixel_translation()
                positions.append(translation.get_pixel_translation())

                if(self.args.loss == "oracle"):
                    remaining_virus_score = batch['locations'].shape[1] - len(positions) 

                else:
                    curr_img = mask_input((1-transformed_mask), curr_img.squeeze(), self.args.masking, self.bg, self.norm_transform).detach().to(DEVICE)
                    model_in = curr_img
                    remaining_virus_score = self.act_fct(self.model(model_in))
                
                iters += 1
            

        positions = np.array(positions)
        num_virus = positions.shape[0]        

        if(SAVE_GRADIENT_IMAGES):
            exit()
        positions, scores, num_virus = self.select_masks_by_iou(positions, batch['capsideradius']*2, batch['image'], self.args.nms_max_iou, individual_masks = True, gt_masks = batch['gt_mask'])
        scores = self.recompute_scores(positions, batch['capsideradius'], batch['image'], batch['gt_mask'])
        return positions, scores, num_virus, training_epochs

    def plot_single_step(self, mask, input_img, translation, gradients, e, iters, title, batch, std):
        plt.close()
        plt.figure()
        
        masked_input = mask_input(mask,input_img,self.args.masking,self.bg, self.norm_transform)
        masked_input = masked_input.squeeze()
        if(len(masked_input.shape)==3):
            masked_input = masked_input.permute(1,2,0)
        
        plt.imshow(min_max(masked_input))
        # plt.imshow(mask, alpha = 0.3)
        pos = translation.get_pixel_translation()
        scores = self.recompute_scores([pos], batch['capsideradius'], batch['image'], batch['gt_mask'])
        plt.title("Current score (maskBG) = "+str(round(title,2))+"\nFinal score (maskOthers) = "+str(scores[0])+"\n"+str(pos))

        grad_x = -1*gradients[0]
        grad_y = -1*gradients[1]

        grad_length = torch.sqrt(grad_x**2 + grad_y**2)
        grad_x = grad_x/grad_length
        grad_y = grad_y/grad_length

        plt.scatter(pos[0], pos[1], color = "red")
        plt.arrow(pos[0], pos[1], grad_x*IMG_SIZE[0]*0.1, grad_y*IMG_SIZE[1]*0.1, color="red", length_includes_head=False, width=3)
        plt.ylim(0,IMG_SIZE[0])
        plt.xlim(0,IMG_SIZE[1])
        plt.axis("off")
        plt.savefig("./DebugImages/Gradients_"+str(iters)+"/"+str(e)+".jpg", dpi = 300)
        plt.close()

    def train_iter(self, batch, curr_img, iters, scaler, num_virus):
        n_forwards = 0 # number of forward passes for initialization           
        if(self.args.initialize == "gradcam"):
            # init pos by CAM
            cam = compute_cam(self.norm_transform(curr_img).to(DEVICE), GradCAM, self.gradcam_model)     
            plot_cam = cam.copy()
        
            pos = self.init_pos_cam(cam)
            translation = TranslationMatrix_iterative(init_pos=pos, gaussian_pdf = self.args.pseudolabels_gaussian_pdf).to(DEVICE)
            if(SAVE_GRADIENT_IMAGES):
                plt.close()
                plt.figure()
                plt.imshow(min_max(curr_img.squeeze().permute(1,2,0)))
                plt.imshow(plot_cam.squeeze(), alpha=0.4)
                pos = translation.get_pixel_translation()
                plt.scatter(pos[0],pos[1], color="red")
                plt.axis('off')
                plt.ylim(0,IMG_SIZE[0])
                plt.xlim(0,IMG_SIZE[1])
                plt.savefig("./DebugImages/Gradients_"+str(iters)+"/init.jpg", dpi = 300)
                plt.close()

                plt.close()
                plt.figure()
                plt.imshow(min_max(curr_img.squeeze().permute(1,2,0)))
                plt.axis('off')
                plt.ylim(0,IMG_SIZE[0])
                plt.xlim(0,IMG_SIZE[1])
                plt.savefig("./DebugImages/Gradients_"+str(iters)+"/image.jpg", dpi = 300)
                plt.close()
        elif(self.args.initialize == "selectivesearch"):
            pos, n_forwards = self.init_pos_selectivesearch(curr_img, batch['capsideradius'])

            if(np.any(pos.numpy() == False)):
                return False, n_forwards
            translation = TranslationMatrix_iterative(init_pos=pos, gaussian_pdf = self.args.pseudolabels_gaussian_pdf).to(DEVICE)
        elif(self.args.initialize == "random"): 
            translation = TranslationMatrix_iterative(gaussian_pdf = self.args.pseudolabels_gaussian_pdf).to(DEVICE)
        elif(self.args.initialize == "oracle"):
            cam = perfectGradCAM(batch['locations'][0], batch['capsideradius'])
        
        optim_t, scheduler_t = self.init_optimizer(translation, self.args.lr_t)

        e = 0
        best_state_dict_t = deepcopy(translation.state_dict())
        curr_val_loss = np.inf
        # start optimization
        while(e <= self.args.max_iters):
            # max iters are reached
            if(e == self.args.max_iters): 
                break

            optim_t.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.args.pseudolabels_use_amp): 
                std = self.start_std*torch.exp(torch.tensor(-1*self.std_fac*e))       
                # std = self.args.std_start - (self.std_fac*e)
                transformed_mask = translation(std.to(DEVICE), batch['capsideradius'].to(DEVICE)) 
                loss = loss_fct(transformed_mask, curr_img, batch['gt_mask'], self.model, (batch['capsideradius']*2).to(DEVICE), self.act_fct, self.bg, self.args, self.norm_transform)     
            scaler.scale(loss).backward()

            scaler.step(optim_t)
            old_scaler = scaler.get_scale()
            scaler.update()
            if((scheduler_t != None) and (self.args.scheduler != "plateau") and (scaler.get_scale() == old_scaler)):
                scheduler_t.step()
            e += 1


            if(SAVE_GRADIENT_IMAGES):
                gradients = translation.translation.grad.cpu()
                loss = loss_fct(transformed_mask, curr_img, batch['gt_mask'], self.model, (batch['capsideradius']*2).to(DEVICE), self.act_fct, self.bg, self.args, self.norm_transform)
                self.plot_single_step(transformed_mask.detach().cpu().detach(), curr_img, translation, gradients, e, iters, -1*loss.item(), batch, std)
         
            # validation
            if((e % self.args.val_step) == 0):
                wandb.log({"optim/lr_t":optim_t.param_groups[0]['lr']}) 
                with torch.no_grad():
                    val_loss, remaining_virus_score, virus_detected_score = self.validation(batch, curr_img, translation, log_wandb = self.args.log_val, num_detections=num_virus)
                if(val_loss < curr_val_loss):
                    best_state_dict_t = deepcopy(translation.state_dict())
                    curr_val_loss = val_loss.item()
                if(self.args.scheduler == "plateau"):
                    scheduler_t.step(val_loss)  
            
        if(self.args.pseudolabels_use_validation):
            translation.load_state_dict(best_state_dict_t)       
        if(SAVE_GRADIENT_IMAGES):
            gradients = torch.Tensor([0,0])
            std = torch.Tensor([0.0]).to(DEVICE)
            transformed_mask = translation(std, batch['capsideradius'].to(DEVICE)) 
            loss = loss_fct(transformed_mask, curr_img, batch['gt_mask'], self.model, (batch['capsideradius']*2).to(DEVICE), self.act_fct, self.bg, self.args, self.norm_transform)
            self.plot_single_step(transformed_mask.detach().cpu().detach(), curr_img, translation, gradients, e, iters, -1*loss.item(), batch, std)
        return translation, e + n_forwards
    