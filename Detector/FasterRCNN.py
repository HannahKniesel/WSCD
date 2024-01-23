import sys
sys.path.insert(0,'../')

from .Detection.faster_rcnn import FastRCNNPredictor
from .Detection.anchor_utils import AnchorGenerator
from .Detection import fasterrcnn_resnet50_fpn
import torch 
from .engine import train_one_epoch, evaluate
from .utils import *
import argparse
import wandb
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from pathlib import Path 
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from Variables import *
from Datasets import *
from Transforms import frcnn_transform_resnet50_training, frcnn_transform_resnet50_inference, frcnn_transform_resnet101_training, frcnn_transform_resnet101_inference


class Detection_FRCNN():
    def __init__(self, args, label_data, seed, timings_path, path_to_training_labels= [], threshold = -1):
        self.args = args
        self.label_data = label_data
        self.threshold = threshold
        deterministic(seed=seed)

        # init 
        if((self.args.backbone == "resnet101") or (self.args.backbone == "vit")): # FasterRCNN is using ResNet101 as backbone
            self.model = fasterrcnn_resnet50_fpn(weights=None, min_size= 224, max_size = 224)
            backbone = resnet_fpn_backbone(backbone_name = "resnet101", weights=ResNet101_Weights.IMAGENET1K_V1)
            # backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.backbone = backbone

            
        elif(self.args.backbone == "resnet50"):
            # for resnet50 use pretrained weights from Conrad, Ryan, and Kedar Narayan. "CEM500K, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning." Elife 10 (2021): e65894.
            self.model = fasterrcnn_resnet50_fpn(weights=None, min_size= 224, max_size = 224, image_mean = (0.58331613,), image_std = (0.09966064,))
            # Load weights from pretrained EM model
            self.model.backbone.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            state_path = EM_PRETRAINED_WEIGHTS#"./../pretrained_models/cem500k_mocov2_resnet50_200ep_pth.tar"
            state = torch.load(state_path, map_location='cpu')
            state_dict = state['state_dict']
            resnet50_state_dict = deepcopy(state_dict)
            for k in list(resnet50_state_dict.keys()):
                #only keep query encoder parameters; discard the fc projection head
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    resnet50_state_dict[k[len("module.encoder_q."):]] = resnet50_state_dict[k]
                #delete renamed or unused k
                del resnet50_state_dict[k]
            self.model.backbone.body.load_state_dict(resnet50_state_dict, strict=False)

        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(DEVICE)

        if(self.args.backbone == "resnet50"):
            frcnn_transform_training = frcnn_transform_resnet50_training
            frcnn_transform_inference = frcnn_transform_resnet50_inference
        elif(self.args.backbone == "resnet101"):
            frcnn_transform_training = frcnn_transform_resnet101_training
            frcnn_transform_inference = frcnn_transform_resnet101_inference

        self.init_logging()
        if(self.args.dataset =="herpes"):
            if(self.label_data=="bb"):
                ds_train = HerpesBBDataset_GT(HERPES_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, self.args.percentage, timings_path, num_imgs = self.args.num_img, preload=self.args.preload)
            elif(self.label_data=="loc"):
                ds_train = HerpesBBDataset_GT(HERPES_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, self.args.percentage, timings_path, num_imgs = self.args.num_img, preload=self.args.preload, loc = True)
            elif(self.label_data=="bin"):
                # data_path = path_to_training_labels overwrites the HERPES_TRAIN_DATA_PATH
                ds_train = HerpesBBDataset_Ours(HERPES_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, self.args.percentage, timings_path, data_paths=path_to_training_labels, threshold = threshold, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_test = HerpesBBDataset_GT(HERPES_TEST_DATA_PATH, frcnn_transform_inference, 42, -1, -1, "", num_imgs = self.args.num_img, preload=self.args.preload)
            ds_val = HerpesBBDataset_GT(HERPES_VAL_DATA_PATH, frcnn_transform_inference, 42, -1, -1, "", num_imgs = self.args.num_img, preload=self.args.preload)
        
        elif(self.args.dataset =="adeno"):
            if(self.label_data=="loc"):
                ds_train = TEMBBDataset_GT(ADENO_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, num_imgs = self.args.num_img, preload=self.args.preload)
            elif(self.label_data=="bin"):
                # data_path = path_to_training_labels overwrites the HERPES_TRAIN_DATA_PATH
                ds_train = TEMBBDataset_Ours(ADENO_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, data_paths=path_to_training_labels, threshold = threshold, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_test = TEMBBDataset_GT(ADENO_TEST_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_val = TEMBBDataset_GT(ADENO_VAL_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
        
        elif(self.args.dataset =="noro"):
            if(self.label_data=="loc"):
                ds_train = TEMBBDataset_GT(NORO_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, num_imgs = self.args.num_img, preload=self.args.preload)
            elif(self.label_data=="bin"):
                # data_path = path_to_training_labels overwrites the HERPES_TRAIN_DATA_PATH
                ds_train = TEMBBDataset_Ours(NORO_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, data_paths=path_to_training_labels, threshold = threshold, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_test = TEMBBDataset_GT(NORO_TEST_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_val = TEMBBDataset_GT(NORO_VAL_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
        
        elif(self.args.dataset =="papilloma"):
            if(self.label_data=="loc"):
                ds_train = TEMBBDataset_GT(PAP_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, num_imgs = self.args.num_img, preload=self.args.preload)
            elif(self.label_data=="bin"):
                # data_path = path_to_training_labels overwrites the HERPES_TRAIN_DATA_PATH
                ds_train = TEMBBDataset_Ours(PAP_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, data_paths=path_to_training_labels, threshold = threshold, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_test = TEMBBDataset_GT(PAP_TEST_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_val = TEMBBDataset_GT(PAP_VAL_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
        
        elif(self.args.dataset =="rota"):
            if(self.label_data=="loc"):
                ds_train = TEMBBDataset_GT(ROT_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, num_imgs = self.args.num_img, preload=self.args.preload)
            elif(self.label_data=="bin"):
                # data_path = path_to_training_labels overwrites the HERPES_TRAIN_DATA_PATH
                ds_train = TEMBBDataset_Ours(ROT_TRAIN_DATA_PATH, frcnn_transform_training, seed, self.args.annotation_time, data_paths=path_to_training_labels, threshold = threshold, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_test = TEMBBDataset_GT(ROT_TEST_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
            ds_val = TEMBBDataset_GT(ROT_VAL_DATA_PATH, frcnn_transform_inference, 42, -1, num_imgs = self.args.num_img, preload=self.args.preload)
        
        

        # define training and validation data loaders
        self.data_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=self.args.frcnn_bs, shuffle=True, num_workers=0,
            collate_fn=collate_fn)

        self.data_loader_test = torch.utils.data.DataLoader(
            ds_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)

        self.data_loader_val = torch.utils.data.DataLoader(
            ds_val, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr = self.args.frcnn_lr) 
        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    def init_logging(self, wandb_run_id=-1):
        if(wandb_run_id < 0):
            
            self.wandb_name = f"FRCNN_Time{self.args.annotation_time}_Perc{self.args.percentage}_{self.label_data}_{self.args.dataset}
            if(self.label_data=="bin"):
                if(self.threshold == -1):
                    self.wandb_name = self.wandb_name+"_probabilities"
                else:
                    self.wandb_name = self.wandb_name+"_F1val"

            os.environ['WANDB_PROJECT']= self.args.project
            wandb.init(config = self.args, reinit=True, group = self.wandb_name, mode=self.args.wandb_mode)
            try:
                self.wandb_name = self.wandb_name+"_"+str(pathlib.Path(self.args.classifier_path).stem)
            except: 
                pass
            wandb_name = self.wandb_name+"_"+str(wandb.run.id)
            wandb.run.name = wandb_name
        else: 
            wandb.init(config = self.args, reinit=True, resume= 'allow', id=wandb_run_id, mode=self.args.wandb_mode)

        # log folder
        log_path = self.args.log_path +"/"+str(wandb.run.project)+"/"+str(wandb.run.name)
        os.makedirs(log_path, exist_ok=True)
        write_txt(log_path+"/args.txt", str(self.args))
        self.log_path = log_path
        print("Log to:")
        print(pathlib.Path(self.log_path).absolute())
        return

    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        curr_map = 0
        model_weights = deepcopy(self.model.state_dict())
        curr_patience = 0
        # for epoch in range(num_epochs):
        iterations = 0
        epoch = 0
        while(iterations < self.args.frcnn_n_iters):
            self.model.train()
            metric_logger = train_one_epoch(self.model, self.optimizer, self.data_loader, scaler, DEVICE, epoch, print_freq=200)

            train_loss = metric_logger.loss.value
            train_loss_classifier = metric_logger.loss_classifier.value
            train_loss_box_reg = metric_logger.loss_box_reg.value
            train_loss_rpn_box_reg = metric_logger.loss_rpn_box_reg.value

            wandb.log({"Training/loss": train_loss}, step = epoch)
            wandb.log({"Training/loss_classifier": train_loss_classifier}, step = epoch)
            wandb.log({"Training/loss_box_reg": train_loss_box_reg}, step = epoch)
            wandb.log({"Training/loss_rpn_box_reg": train_loss_rpn_box_reg}, step = epoch)
            self.lr_scheduler.step()

            epoch +=1
            iterations += len(self.data_loader.dataset)

            # validation
            self.model.eval()
            self.visualize_validation(self.data_loader_val)
            metric_logger = evaluate(self.model, self.data_loader_val, device=DEVICE)
            for iou_type, coco_eval in metric_logger.coco_eval.items():
                mean_ap = coco_eval.stats[0].item()
                mean_ap_50 = coco_eval.stats[1].item()
                mean_ap_75 = coco_eval.stats[2].item()

            wandb.log({"Val/mAP": mean_ap})
            wandb.log({"Val/mAP50": mean_ap_50})
            wandb.log({"Val/mAP75": mean_ap_75})
            wandb.log({"Val/Epoch": epoch})



            if(mean_ap_50 > curr_map):
                curr_map = mean_ap_50
                model_weights = deepcopy(self.model.state_dict()) # save best state dict
                # Save parameters
                dict_saveparams = {}
                set_param(self.data_loader.dataset.path, 'training_paths', dict_saveparams)
                set_param(self.data_loader.dataset.percentage, 'training_data_size', dict_saveparams)
                set_param(self.args.annotation_time, 'annotation_time', dict_saveparams)

                set_param(wandb.run.id, 'wandb_id', dict_saveparams)
                set_param(wandb.run.name, 'wandb_name', dict_saveparams)
                set_param(wandb.run.group, 'wandb_group', dict_saveparams)
                set_param(wandb.run.project, 'wandb_project', dict_saveparams)
                set_param(wandb.run.get_url(), 'wandb_url', dict_saveparams)
                set_param(self.model.state_dict(), 'model', dict_saveparams)
                set_param(self.args.backbone, 'backbone', dict_saveparams)
                
                save_dict(dict_saveparams, self.log_path+"/training_state.pth")
                curr_patience = 0
            else:
                curr_patience += 1
            
            if((curr_patience > PATIENCE_EARLY_STOPPING) and (iterations > MIN_ITERATIONS)):
                break

        # load best validation weights
        self.model.load_state_dict(model_weights)
        metric_logger = evaluate(self.model, self.data_loader_test, device=DEVICE)
        for iou_type, coco_eval in metric_logger.coco_eval.items():
            mean_ap = coco_eval.stats[0].item()
            mean_ap_50 = coco_eval.stats[1].item()
            mean_ap_75 = coco_eval.stats[2].item()

        wandb.log({"Test/mAP": mean_ap})
        wandb.log({"Test/mAP50": mean_ap_50})
        wandb.log({"Test/mAP75": mean_ap_75})

        # save final model 
        dict_saveparams = {}
        set_param(self.data_loader.dataset.path, 'training_paths', dict_saveparams)
        set_param(self.data_loader.dataset.percentage, 'training_data_size', dict_saveparams)
        set_param(self.args.annotation_time, 'annotation_time', dict_saveparams)

        set_param(wandb.run.id, 'wandb_id', dict_saveparams)
        set_param(wandb.run.name, 'wandb_name', dict_saveparams)
        set_param(wandb.run.group, 'wandb_group', dict_saveparams)
        set_param(wandb.run.project, 'wandb_project', dict_saveparams)
        set_param(wandb.run.get_url(), 'wandb_url', dict_saveparams)

        set_param(self.model.state_dict(), 'model', dict_saveparams)
        set_param(self.args.backbone, 'backbone', dict_saveparams)
        set_param(mean_ap, 'map', dict_saveparams)
        set_param(mean_ap_50, 'map50', dict_saveparams)
        set_param(mean_ap_75, 'map75', dict_saveparams)

        save_dict(dict_saveparams, self.log_path+"/training_state.pth")

    # only works for batch size of 1
    def predict(self, input_tensor, model, detection_threshold):
        names = ['__background__', 'virus']
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        outputs = model(input_tensor)
        pred_classes = [names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

        boxes, classes, labels, indices = [], [], [], []
        for index in range(len(pred_scores)):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)

        boxes = np.int32(boxes)
        return boxes, classes, labels, indices

    # generate image from bb coordinates
    def make_bb(self, boxes):
        full_boxes = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        for box in boxes: 
            xmin, ymin, xmax, ymax = box
            full_boxes[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        return full_boxes

    def visualize_validation(self, dataloader):
        num_imgs = 5
        bb_pred = []
        bb_gt = []
        imgs = []
        for idx, (images, targets) in enumerate(dataloader):
            for i,t in zip(images, targets):
                boxes_true = t['boxes'].numpy()
                if(boxes_true.shape == (0,4)):
                    continue
                
                boxes_pred, classes, labels, indices = self.predict(i, self.model, 0.5) 
                img = i.numpy().squeeze()
                
                # get bb
                boxes_true = t['boxes'].numpy()
                boxes_true = self.make_bb(boxes_true)
                boxes_pred = self.make_bb(boxes_pred)
                
                bb_pred.append(boxes_pred)
                bb_gt.append(boxes_true)
                imgs.append(img)
            if(len(imgs) == num_imgs):
                break
        
        plt.close()
        fig, axs = plt.subplots(3,num_imgs, figsize=(num_imgs*5, 5))
        for i in range(num_imgs):
            img = imgs[i]
            full_boxes = bb_pred[i]
            gt_box = bb_gt[i]
            
            if(len(img.shape)==3):
                img = img.transpose(1,2,0) 
            axs[0,i].imshow((img))
            axs[0,i].set_axis_off()
            axs[0,i].set_title("Input\nMin: "+str(np.min(img))+" Max: "+str(np.max(img)))
            axs[1,i].imshow((full_boxes))
            axs[1,i].set_axis_off()
            axs[1,i].set_title("Pred Boxes\nMin: "+str(np.min(gt_box))+" Max: "+str(np.max(full_boxes)))
            axs[2,i].imshow((gt_box))
            axs[2,i].set_axis_off()
            axs[2,i].set_title("GT Boxes\nMin: "+str(np.min(gt_box))+" Max: "+str(np.max(gt_box)))
            
        fig.tight_layout()
        wandb.log({"Plot "+str(0.5): wandb.Image(plt)})
        plt.close() 
        return      
