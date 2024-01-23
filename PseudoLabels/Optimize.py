import argparse
import wandb
import torch
import os
import numpy as np
import pathlib
from torch.utils.data import DataLoader
from copy import deepcopy
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import matplotlib.patches as patches



import sys
sys.path.insert(0,'..')
sys.path.insert(0, '../Detector/')
from Detector.engine import evaluate_pseudo
from Utils import *
from Utils_Eval import *
from Variables import *
from Datasets import HerpesLabelGeneration_Dataset, TEMLabelGeneration_Dataset
from Transforms import pseudolabels_transform_resnet101, pseudolabels_transform_resnet50, norm_resnet101, norm_resnet50
from GradCAM import GradCAM

def loss_fct(transformed_mask, input_img, gt_mask, model, capsid_size, act_fct, bg, args, norm_transform):    

    if(args.loss == "score"):
        # maximize score where mask shows virus
        model_in = mask_input(transformed_mask, input_img, args.masking, bg, norm_transform)
        score_virus = torch.mean(act_fct(model(model_in)))

    elif(args.loss == "logit"):
        # maximize score where mask shows virus
        model_in = mask_input(transformed_mask, input_img, args.masking, bg, norm_transform)
        score_virus = torch.mean(model(model_in))

    elif(args.loss == "oracle"):
        score_virus = model(transformed_mask, gt_mask, capsid_size/2)
            
    loss = -1*score_virus 
    return loss

# shows all masks in R channel of RGB weighted by their score
def show_all_masks(transformed_mask, model, input_img, masking, bg, act_fct, norm_transform, capsid_size, loss = "none", gt_mask = None):
    model_in = mask_input(transformed_mask, input_img, masking, bg, norm_transform)

    if(loss == "oracle"):
        score = model(transformed_mask, gt_mask, capsid_size/2).unsqueeze(0)
    else: 
        logit = model(model_in)
        score = act_fct(logit)
    score_fac = torch.clip(score+0.3, 0, 1)
    combined_mask = torch.max((score_fac[:,:,None,None]*transformed_mask), dim=0)[0].unsqueeze(0) 
    return combined_mask
 

class Optimizer():
    # model with identyty function as final activation function
    # log path = model path?
    def __init__(self, args, log_path, data_split, title, model, gradcam_model, data_paths, seed):
        self.log_path = log_path
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        self.data_split = data_split

        self.args = args
        self.num_data = self.args.num_img

        
        self.metric = MeanAveragePrecision() 
        self.act_fct = torch.nn.Sigmoid()
        self.init_logging(title)
    
        self.model = model
        self.model.to(DEVICE)
        self.model.eval()     

        self.gradcam_model = gradcam_model
        self.gradcam_model.to(DEVICE)
        self.gradcam_model.eval()   
        for param in self.gradcam_model.parameters():
            param.requires_grad = True

        # don't optimize model parameters
        for param in self.model.parameters():
            param.requires_grad = False   

        if(self.args.dataset == "herpes"):
            if(data_split == "val"):
                self.data_p = HERPES_VAL_DATA_PATH
            elif(data_split == "test"):
                self.data_p = HERPES_TEST_DATA_PATH
            elif(data_split == "train"):
                self.data_p = HERPES_TRAIN_DATA_PATH

        elif(self.args.dataset == "adeno"):
            if(data_split == "val"):
                self.data_p = ADENO_VAL_DATA_PATH
            elif(data_split == "test"):
                self.data_p = ADENO_TEST_DATA_PATH
            elif(data_split == "train"):
                self.data_p = ADENO_TRAIN_DATA_PATH

        elif(self.args.dataset == "noro"):
            if(data_split == "val"):
                self.data_p = NORO_VAL_DATA_PATH
            elif(data_split == "test"):
                self.data_p = NORO_TEST_DATA_PATH
            elif(data_split == "train"):
                self.data_p = NORO_TRAIN_DATA_PATH

        elif(self.args.dataset == "papilloma"):
            if(data_split == "val"):
                self.data_p = PAP_VAL_DATA_PATH
            elif(data_split == "test"):
                self.data_p = PAP_TEST_DATA_PATH
            elif(data_split == "train"):
                self.data_p = PAP_TRAIN_DATA_PATH

        elif(self.args.dataset == "rota"):
            if(data_split == "val"):
                self.data_p = ROT_VAL_DATA_PATH
            elif(data_split == "test"):
                self.data_p = ROT_TEST_DATA_PATH
            elif(data_split == "train"):
                self.data_p = ROT_TRAIN_DATA_PATH
        
        
        print("INFO:: Use device "+str(DEVICE))

        # get background data for inpainting
        if(self.args.backbone == "resnet50"):
            transform = pseudolabels_transform_resnet50
            self.norm_transform = norm_resnet50
            self.means = torch.tensor([0.58331613])[None, :, None, None]
            self.stds = torch.tensor([0.09966064])[None, :, None, None]     

        elif((self.args.backbone == "resnet101") or (self.args.backbone == "vit")):
            transform = pseudolabels_transform_resnet101
            self.norm_transform = norm_resnet101
            self.means = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            self.stds = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]     


        self.bg = HerpesLabelGeneration_Dataset(HERPES_TRAIN_DATA_PATH, transform, 42, -1, -1, "", preload= True, num_virus=0, num_imgs = 100)
        if(self.args.dataset == "herpes"):
            self.ds_train = HerpesLabelGeneration_Dataset(self.data_p, transform, seed, -1, -1, CLASSIFICATION_TIMINGS, corrupt_size= self.args.corrupt_size, data_paths=data_paths, preload= self.args.preload, num_imgs=self.num_data, num_virus=self.args.num_virus, start_idx = self.args.start_idx)
        else: 
            self.ds_train = TEMLabelGeneration_Dataset(self.data_p, transform, seed, -1, data_paths=data_paths, preload= self.args.preload, num_imgs=self.num_data, num_virus=self.args.num_virus, start_idx = self.args.start_idx)

    def init_logging(self, title):
        try: 
            wandb.finish()
        except:
            print("WARNING::Could not finish previous run.")
            pass
        
        self.wandb_name = title+str(self.args.masking)+"_"+str(self.data_split)+"_"+str(self.args.loss)+"_"+str(self.args.lr_t)+"_"+str(self.args.lr_t_final)+"_"+str(self.args.dataset)
        self.wandb_name = self.wandb_name+"_cam"+str(self.args.initialize)
        if(self.args.annotation_time>0):
            self.wandb_name = self.wandb_name+"_"+str(self.args.annotation_time)
        if(self.args.percentage>0):
            self.wandb_name = self.wandb_name+"_"+str(self.args.percentage)
        
        if(title == "Iterative_"):
            self.wandb_name = self.wandb_name+"_iters"+str(self.args.max_iters)
        elif(title == "SelectiveSearch_"):
            self.wandb_name = self.wandb_name+"_"+str(self.args.selective_search_mode)+"_"+str(self.args.selective_search_topn)
        else: 
            self.wandb_name = self.wandb_name+"_step"+str(self.args.step)
            
        self.wandb_name = self.wandb_name+"_"+str(self.args.scheduler)+"_"+str(self.args.std_end)

        os.environ['WANDB_PROJECT']= self.args.project
        wandb.init(config = self.args, reinit=True, group = self.wandb_name, mode=self.args.wandb_mode)
        wandb_name = self.wandb_name+"_"+str(wandb.run.id)
        self.wandb_name = self.wandb_name+"_"+str(pathlib.Path(self.args.classifier_path).stem)

        wandb.run.name = wandb_name

        self.save_to = self.log_path+str(wandb.run.name)+"/"
        if(self.args.save_data):
            print("Save data to: "+str(self.save_to))  
            os.makedirs(self.save_to, exist_ok=True)
            write_txt(self.save_to+"/args.txt", str(self.args))
        return

    def init_optimizer(self, model, lr):
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0, momentum = self.args.momentum)
        if(self.args.scheduler == "None"):
            scheduler = None
        elif(self.args.scheduler == "cos"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = int(self.args.max_iters), eta_min = lr*self.args.lr_t_final)
        elif(self.args.scheduler == "exp"):
            scheduler  = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = 0.5)
        elif(self.args.scheduler == "step"):
            scheduler  = torch.optim.lr_scheduler.StepLR(optim, step_size = 5, gamma = 0.5)
        elif(self.args.scheduler == "plateau"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
        return optim, scheduler


    def get_std(self, capsideradius=-1):
        max_iters = self.args.max_iters
        if(capsideradius):
            start_std = self.args.std_start * (capsideradius/IMG_SIZE[0])
            final_std = self.args.std_end * (capsideradius/IMG_SIZE[0]) # 68% of data inside gaussian for radius = std; 99% of data is inside gaussian for 2*radius = std
        else:
            final_std = self.args.std_end #0.002
            start_std = self.args.std_start
        x = -1*(np.log(final_std/start_std)/max_iters)
        smallest_max_gradient = gaussian_gradient_torch(start_std, 0, start_std).numpy()[0] # use to set initial scaling to 1 --> hence the learning rate remains the same.
        return x, start_std, smallest_max_gradient
    
    def recompute_scores(self, positions, capside_radius, input_img, gt_mask = [None]*50):
        # get BB scores                                            
        transformed_mask = generate_masks_from_positions(positions, capside_radius.numpy())    
        if(self.args.score_bb == "mask_other_virus"):
            virus_masks = []
            if(transformed_mask.shape[0]>1):
                for i in range(transformed_mask.shape[0]):
                    virus_mask = transformed_mask.clone()
                    virus_mask[i,...] = 0
                    virus_mask = torch.max(virus_mask, axis = 0)[0]
                    virus_mask = 1- virus_mask
                    virus_masks.append(virus_mask)
                virus_masks = torch.concat(virus_masks)
                masks = virus_masks[:,None,:,:]
            else: 
                masks = torch.ones((1,1,IMG_SIZE[0], IMG_SIZE[1]))
        else: 
            masks = transformed_mask
        score = self.get_scores_batchified(masks, input_img, self.model, (capside_radius*2).to(DEVICE), gt_masks = gt_mask)
        return score

    def get_scores_batchified(self, masks, input_img, model, capsid_size, gt_masks):
        idx = 0
        all_scores = []
        while(idx < masks.shape[0]):
            if(idx+BATCH_SIZE < masks.shape[0]):
                curr_masks = masks[idx:idx+BATCH_SIZE]
                curr_gts = gt_masks[idx:idx+BATCH_SIZE]
            else: 
                curr_masks = masks[idx:]
                curr_gts = gt_masks[idx:]

            idx = idx+BATCH_SIZE
            
            if(self.args.loss == "oracle"):
                scores = np.ones((len(curr_masks),)) 
            else: 
                model_in = mask_input(curr_masks, input_img, self.args.masking, self.bg, self.norm_transform).to(DEVICE)
                scores = self.act_fct(model(model_in)).detach().cpu().numpy()[:,0]

            all_scores.extend(scores)
        return all_scores
    

    # remove masks with higher IOU than MAX_IOU 
    def select_masks_by_iou(self, positions, capside_diameter, input_img, max_iou, individual_masks = False, gt_masks = [None]*50): 
        # get masks for iou computation and their relating scores
        transformed_mask = generate_masks_from_positions(positions, float(capside_diameter/2))    
        score = self.get_scores_batchified(transformed_mask, input_img, self.model, capside_diameter.to(DEVICE), gt_masks) 
        
        check_masks = [mask for mask in transformed_mask]
        masks = []
        save_scores = []
        save_positions = []        

        while(len(check_masks)>0): 
            new_score = []
            new_check_masks = []
            new_positions = []
            idx_max = np.argmax(score) 
            max_mask = check_masks[idx_max]
            masks.append(max_mask) # add mask to list        
            save_positions.append(positions[idx_max])
            save_scores.append(score[idx_max])

            for i in range(len(check_masks)): 
                mask = check_masks[i]
                if(i == idx_max):
                    continue
                else: 
                    iou_val = iou(mask, max_mask)
                    if(iou_val<max_iou):
                        new_check_masks.append(mask)
                        new_score.append(score[i].squeeze())
                        new_positions.append(positions[i])

            score = torch.tensor(new_score)
            check_masks = new_check_masks 
            positions = new_positions
        num_virus = len(masks)
        if(individual_masks):
            return np.array(save_positions), np.array(save_scores), num_virus
        return np.array(save_positions), np.array(save_scores), num_virus
    
    
    
    
    def get_bboxes_from_positions(self, positions, scores, radius):
        if(np.any(positions == -1)):
            bboxes = np.array([]).reshape(-1, 4)
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            preds = [dict(boxes=bboxes, scores=torch.tensor([]), labels=torch.tensor([], dtype=torch.int64))]
        else: 
            scores_new = []
            bboxes = []
            for pos, score in zip(positions, scores):    
                scores_new.append(score)   
                x = pos[0]
                y = pos[1]

                xmin = np.max((0, x-radius))
                xmax = np.min((IMG_SIZE[0], x+radius))        
                ymin = np.max((0, y-radius))
                ymax = np.min((IMG_SIZE[0], y+radius))   
                box = [float(xmin),float(ymin),float(xmax),float(ymax)]

                bboxes.append(box)
            preds = [dict(boxes=torch.tensor(bboxes).type(torch.float), scores=torch.from_numpy(np.array(scores_new)), labels=torch.tensor([1.], dtype=torch.int64).repeat(len(bboxes)))]
        return preds
    
    def get_bboxes_from_masks(self, individual_masks, scores):
        # doesn't contain bounding box
        if(individual_masks.max() == individual_masks.min()):
            bboxes = np.array([]).reshape(-1, 4)
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            preds = [dict(boxes=bboxes, scores=torch.tensor([]), labels=torch.tensor([], dtype=torch.int64))]

        else:
            scores_new = []
            bboxes = []
            for mask, score in zip(individual_masks, scores):    
                if(np.sum(mask)==0):
                    continue  
                scores_new.append(score)           
                mask = mask.squeeze()
                xmask = np.max(mask, axis = 0)
                ymask = np.max(mask, axis = 1)

                xidx = np.argwhere(xmask == 1)
                xmin = np.min(xidx)
                xmax = np.max(xidx)

                yidx = np.argwhere(ymask == 1)
                ymin = np.min(yidx)
                ymax = np.max(yidx)
                box = [xmin,ymin,xmax,ymax]
                # box = [ymin,xmin,ymax,xmax]

                bboxes.append(box)
            preds = [dict(boxes=torch.tensor(bboxes).type(torch.float), scores=torch.from_numpy(np.array(scores_new)), labels=torch.tensor([1.]).repeat(len(bboxes)))]
        return preds  
    

    @torch.no_grad()
    def validation(self, batch, curr_img, translation, log_wandb = True, num_detections = 0):
        std = torch.Tensor([0.0]).to(DEVICE)
        transformed_mask = translation(std, batch['capsideradius'].to(DEVICE))
        val_loss = loss_fct(transformed_mask, curr_img, batch['gt_mask'], self.model, (batch['capsideradius']*2).to(DEVICE), self.act_fct, self.bg, self.args, self.norm_transform)
        
        mask_virus = mask_input((1-transformed_mask), curr_img.squeeze().to(DEVICE), self.args.masking, self.bg, self.norm_transform)
        mask_bg = mask_input((transformed_mask), curr_img.squeeze().to(DEVICE), self.args.masking, self.bg, self.norm_transform)
    
        if(self.args.loss == "oracle"):
            remaining_virus_score = batch['locations'].shape[1] - num_detections 
            virus_detected_score = self.model(transformed_mask, batch['gt_mask'], batch['capsideradius'])
        else: 
            remaining_virus_score = self.act_fct(self.model(mask_virus))
            virus_detected_score = self.act_fct(self.model(mask_bg))

        if(log_wandb):
            wandb.log({'val/Loss': val_loss})
            wandb.log({'val/remaining_virus_score': remaining_virus_score})
            wandb.log({'val/virus_detected_score': virus_detected_score})

            masks = show_all_masks(transformed_mask.to(DEVICE), self.model, curr_img.to(DEVICE), self.args.masking, self.bg, self.act_fct, self.norm_transform, (batch['capsideradius']*2).to(DEVICE), loss = self.args.loss, gt_mask = batch['gt_mask'])
            plt.close()
            plt.clf()
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(min_max(curr_img[0].detach().cpu().squeeze()))
            axs[1].imshow(masks.detach().cpu().squeeze())
            plt.tight_layout()
            wandb.log({"val/img ": wandb.Image(plt)})
            plt.close(fig)
        return val_loss, remaining_virus_score, virus_detected_score

    @torch.no_grad()
    def evaluate(self, predictions, image_ids):
        data_loader = DataLoader(self.ds_train, batch_size=1, shuffle=False, drop_last=False, num_workers = 0)
        coco_evaluator = evaluate_pseudo(predictions, image_ids, data_loader)
        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            mAP = coco_eval.stats[0].item()
            mAP_50 = coco_eval.stats[1].item()
            mAP_75 = coco_eval.stats[2].item()
        return mAP, mAP_50, mAP_75

    def plot_result(self, batch, pred, target, individual_masks, result, hasvirus):
        input_img = batch['image']
        # num_virus = len(target[0]['boxes'])

        # show final detected virus
        masks = show_all_masks(torch.from_numpy(individual_masks).to(DEVICE), self.model, input_img.to(DEVICE), self.args.masking, self.bg, self.act_fct, self.norm_transform, (batch['capsideradius']*2).to(DEVICE), loss = self.args.loss, gt_mask = batch['gt_mask'])
        masks = masks.detach().cpu().numpy().squeeze()
        plt.close()
        plt.clf()
        cols = 3
        rows = 1
        fig,axs = plt.subplots(rows, cols, figsize=(4*cols, 4))
        plt.suptitle("mAP = "+str(round(float(result['map'])*100,2))+" mAP50 = "+str(round(float(result['map_50'])*100,2))+" mAP75 = "+str(round(float(result['map_75'])*100,2)))
        # gt = batch['gt_mask'].cpu().numpy().squeeze()
        gt_pos = []
        for box in target[0]['boxes']:
            xmin,ymin,xmax,ymax = box
            pos = [float(xmin+((xmax-xmin)/2)), float(ymin+((ymax-ymin)/2))]
            gt_pos.append(pos)

        if(len(gt_pos)>0):
            gt_mask = generate_masks_from_positions(gt_pos, batch['capsideradius'].numpy()).max(dim=0)[0].squeeze()
        else:
            gt_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))

        img = min_max(batch['image'].cpu().numpy().squeeze())
        if(len(img.shape)==3):
            axs[0].imshow(img.transpose(1,2,0))
        else:
            axs[0].imshow(img) 
        
        for box in pred[0]['boxes']:
            xmin,ymin,xmax,ymax = box
            rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=2, edgecolor='r', facecolor='none')
            axs[0].add_patch(rect)
        for box in target[0]['boxes']:
            xmin,ymin,xmax,ymax = box
            rect = patches.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), linewidth=2, edgecolor='g', facecolor='none')
            axs[0].add_patch(rect)


        gt_pred_all = np.concatenate((masks[:,:,None],gt_mask[:,:,None],np.zeros_like(masks[:,:,None])), -1)
        axs[1].imshow(gt_pred_all)

        if(self.args.initialize == "oracle"):
            cam = perfectGradCAM(batch['locations'][0], batch['capsideradius'])
        else:
            cam = compute_cam(self.norm_transform(batch['image']).to(DEVICE), GradCAM, self.gradcam_model)
            
        axs[2].imshow(cam.squeeze())
        axs[0].set_title("Bounding Boxes\nGT with annotated virus size")
        axs[1].set_title("Optimizations\nGT with mean virus size")
        axs[2].set_title("GradCAM")

        for ax in axs: 
            ax.set_axis_off()
        plt.tight_layout()
        if(hasvirus!= 0):
            wandb.log({"final/img_hasvirus": wandb.Image(plt)})
        else:
            wandb.log({"final/img_novirus": wandb.Image(plt)})
        plt.close(fig)
        return 


    def train(self):
        dl_train = DataLoader(self.ds_train, batch_size=1, shuffle=False, drop_last=False, num_workers = 0)
        num_positives = 0
        
        # accumulate metrics in list
        results_iou = np.zeros((len(dl_train),)) -1
        time_delta_accumulated = 0

        optimization_steps_lst = []
        

        for i,(batch, target) in enumerate(dl_train):
            if((self.data_split != "train") and (self.args.loss != "oracle")):
                # if datasplit is test or val, use predictions
                model_in = self.norm_transform(batch['image']).to(DEVICE)
                prediction = self.act_fct(self.model(model_in)).detach()
            else: 
                # if datasplit is train or oracle classifier use label
                prediction = batch['label'] 

            # find virus particles
            if(prediction >= self.args.pseudolabel_threshold):      
                num_positives += 1  
                wandb.log({"data/area": torch.mean(batch['gt_mask'])})
                wandb.log({"data/num_virus": batch['label']})
                wandb.log({"data/idx": i})
                start_time = time.time()

                # optimize one image by translation
                positions, scores, num_virus, optimization_steps = self.train_translation(batch, i)

                
                if(np.any(positions[0]==-1)):
                    # did not find one virus
                    individual_masks = np.zeros((1,1,IMG_SIZE[0], IMG_SIZE[1])).astype(np.int32)
                else:
                    # recompute masks with expected capside 
                    individual_masks = generate_masks_from_positions(positions, batch['capsideradius'].numpy())
                    individual_masks = individual_masks.detach().cpu().numpy()
                
                end_time = time.time()
                optimization_steps_lst.append(optimization_steps)
                wandb.log({"Steps/current": optimization_steps})


            # no virus particles have been detected. 
            else: 
                start_time = time.time()
                individual_masks = np.zeros((1,1,IMG_SIZE[0], IMG_SIZE[1])).astype(np.int32)
                positions = np.array([-1,-1]).astype(np.int16)
                scores = np.array([])
                num_virus = 0
                end_time = time.time()

        
            # compute IOU of mask
            pred_single_mask_torch = torch.from_numpy(np.max(individual_masks, axis = 0).squeeze())
            gt_single_mask_torch = torch.from_numpy(np.concatenate(batch['gt_mask'].cpu().numpy()))
            iou_value = iou(pred_single_mask_torch, gt_single_mask_torch)
            results_iou[i] = (iou_value)
            
            # get bounding boxes for the mask
            preds = self.get_bboxes_from_positions(positions, scores, batch['capsideradius'].numpy())
            targets = [dict(boxes=target['boxes'][0].type(torch.float), labels=target['labels'][0])]
            result = self.metric(preds, targets)
            
            # plot img when targets or predictions are not empty.
            if((preds[0]['boxes'].shape[0]>0) or (targets[0]['boxes'].shape[0]>0)):
                hasvirus = batch['label']
                self.plot_result(batch, preds, targets, individual_masks, result, hasvirus)

            
            # time per image
            time_delta = end_time - start_time
            time_delta_accumulated += time_delta
            wandb.log({"data/avg_time": time_delta_accumulated/(i+1)})
            print("INFO::Average time for one image = "+str(time_delta_accumulated/(i+1))+"s | Image "+str(i)+"/"+str(len(dl_train)))

            # save data
            if(self.args.save_data):
                save_as_pickle([IMG_SIZE,
                                positions,
                                scores,
                                batch['capsideradius'].numpy(),
                                float(prediction),
                                time_delta,
                                str(batch['path']), 
                                float(iou_value), 
                                min_max(batch['image']), # apply minmax normalization in order to remove transform based normalization.
                                target['boxes'],
                                target['labels'],
                                self.args.classifier_path+"/training_state.pth"], self.save_to+str(num_virus)+"_"+str(i))
                print("Saved to: "+str(self.save_to+str(num_virus)+"_"+str(i)+".pkl"))
        
        result = self.metric.compute()
        
        # log final metrics
        print("INFO::Pseudolabels of dataset \n Current mAP_50 = "+str(result['map_50'])
                    +"\nCurrent mAP_75 = "+str(result['map_75'])
                    +"\nCurrent mAP = "+str(result['map'])
                    +"\nCurrent mIOU = "+str(np.mean(results_iou[results_iou>=0])))        
        wandb.log({"Test/mIOU": np.mean(results_iou[results_iou>=0])})
        wandb.log({"Test/mAP50": result['map_50']})
        wandb.log({"Test/mAP75": result['map_75']})
        wandb.log({"Test/mAP": result['map']})

        wandb.log({"Steps/mean": np.mean(optimization_steps_lst)})

        dict_saveparams = {}
        set_param(self.ds_train.path, 'training_paths', dict_saveparams)
        set_param(self.ds_train.percentage, 'training_data_size', dict_saveparams)
        set_param(self.args.annotation_time, 'annotation_time', dict_saveparams)
        set_param(wandb.run.id, 'wandb_id', dict_saveparams)
        set_param(wandb.run.name, 'wandb_name', dict_saveparams)
        set_param(wandb.run.group, 'wandb_group', dict_saveparams)
        set_param(wandb.run.project, 'wandb_project', dict_saveparams)
        set_param(wandb.run.get_url(), 'wandb_url', dict_saveparams)
        set_param(self.args.backbone, 'backbone', dict_saveparams)
        set_param(result['map'], 'map', dict_saveparams)
        set_param(result['map_75'], 'map75', dict_saveparams)
        set_param(result['map_50'], 'map50', dict_saveparams)
        set_param(results_iou[results_iou>=0], 'mIOU', dict_saveparams)

        c_path, c_project, c_group, c_url = get_classifier_wandb_params(self.args.classifier_path+"/training_state.pth")
        wandb.log({'classifier':c_url}) 

        set_param(c_path, 'classifier_path', dict_saveparams)
        set_param(c_project, 'classifier_project', dict_saveparams)
        set_param(c_group, 'classifier_group', dict_saveparams)
        set_param(c_url, 'classifier_url', dict_saveparams)
        save_dict(dict_saveparams, self.save_to+"/results.pth")

        if(self.data_split =="train"):
            path_to_training_labels = self.save_to
        else: 
            path_to_training_labels = None
        return path_to_training_labels


    
