import os
import time

import torch
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torchvision.models as models
from pathlib import Path
from GradCAM import GradCAM
import torch.nn.functional as F

from Datasets import Herpes_Classification, TEM_Classification
from Transforms import classification_transform_resnet50_training, classification_transform_resnet50_inference, classification_transform_resnet101_training, classification_transform_resnet101_inference
from Variables import *
from Utils import *
from Utils_Eval import *
from .Models import Classifier_ResNet50, Classifier_ResNet101, Classifier_ViT


sig = torch.nn.Sigmoid()

cam_dict = {"gradcam":GradCAM}
cam_dict_name = {"gradcam":"GradCAM"}

class Training():
    def __init__(self, args, sweep, seed):
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        self.args = args
        print("INFO:: Use device "+str(DEVICE))

        # wandb
        if(not sweep):
            os.environ['WANDB_PROJECT']= args.project        
        name = f"{self.wandb_name}_{args.dataset}_Time_{self.args.annotation_time}_Perc_{self.args.percentage}
        wandb.init(config = args, reinit=True, group = name, mode = self.args.wandb_mode)
        wandb_name = f"{seed}_{name}_{wandb.run.id}
        wandb.run.name = wandb_name
        wandb.run.save()

        # log folder
        log_path = self.args.log_path +"/"+str(wandb.run.project)+"/"+str(wandb.run.name)
        os.makedirs(log_path, exist_ok=True)
        write_txt(log_path+"/args.txt", str(args))

        self.log_path = log_path

    def init_model(self, final_act):
        if(self.args.backbone == "resnet50"):
                model = Classifier_ResNet50()
        elif(self.args.backbone == "resnet101"): 
                model = Classifier_ResNet101()
        elif(self.args.backbone == "vit"):
            model = Classifier_ViT()
        model = torch.nn.Sequential(model, final_act)  
        model.to(DEVICE)
        return model


    def init_optim(self, model):
        if(self.args.classifier_optim =="sgd"):
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.classifier_lr, momentum=0.9)
        elif(self.args.classifier_optim =="adam"):
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.classifier_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return optimizer, scheduler


class TrainingClassifier(Training):
    
    def __init__(self, args, sweep = False, seed=42):
        self.wandb_name = "Classifier_"
        super().__init__(args, sweep, seed)  
        if(self.args.backbone == "resnet50"):
            transform_training = classification_transform_resnet50_training
            transform_inference = classification_transform_resnet50_inference

        elif((self.args.backbone == "resnet101") or (self.args.backbone =="vit")):
            transform_training = classification_transform_resnet101_training
            transform_inference = classification_transform_resnet101_inference

        if(self.args.dataset == "herpes"):
            ds_train = Herpes_Classification(HERPES_TRAIN_DATA_PATH, transform_training, seed, self.args.annotation_time, self.args.percentage, CLASSIFICATION_TIMINGS, num_data=self.args.num_img, preload=self.args.preload)
            ds_test = Herpes_Classification(HERPES_TEST_DATA_PATH, transform_inference, 42, -1, -1, "", num_data=self.args.num_img, preload=self.args.preload)
            ds_val = Herpes_Classification(HERPES_VAL_DATA_PATH, transform_inference, 42, -1, -1, "", num_data=self.args.num_img, preload=self.args.preload)
        elif(self.args.dataset == "adeno"):
            ds_train = TEM_Classification(ADENO_TRAIN_DATA_PATH, transform_training, seed, self.args.annotation_time, num_data=self.args.num_img, preload=self.args.preload)
            ds_test = TEM_Classification(ADENO_TEST_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
            ds_val = TEM_Classification(ADENO_VAL_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
        elif(self.args.dataset == "noro"):
            ds_train = TEM_Classification(NORO_TRAIN_DATA_PATH, transform_training, seed, self.args.annotation_time, num_data=self.args.num_img, preload=self.args.preload)
            ds_test = TEM_Classification(NORO_TEST_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
            ds_val = TEM_Classification(NORO_VAL_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
        elif(self.args.dataset == "papilloma"):
            ds_train = TEM_Classification(PAP_TRAIN_DATA_PATH, transform_training, seed, self.args.annotation_time, num_data=self.args.num_img, preload=self.args.preload)
            ds_test = TEM_Classification(PAP_TEST_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
            ds_val = TEM_Classification(PAP_VAL_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
        elif(self.args.dataset == "rota"):
            ds_train = TEM_Classification(ROT_TRAIN_DATA_PATH, transform_training, seed, self.args.annotation_time, num_data=self.args.num_img, preload=self.args.preload)
            ds_test = TEM_Classification(ROT_TEST_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
            ds_val = TEM_Classification(ROT_VAL_DATA_PATH, transform_inference, 42, -1, num_data=self.args.num_img, preload=self.args.preload)
        

        self.n_imgs_train = len(ds_train)
        self.n_imgs_test = len(ds_test)
        self.n_imgs_val = len(ds_val)

        self.training_loader = DataLoader(ds_train, batch_size=self.args.classifier_bs, shuffle=True, drop_last = True)
        self.validation_loader = DataLoader(ds_val, batch_size=self.args.classifier_bs*2, shuffle=False, drop_last = False)
        self.test_loader = DataLoader(ds_test, batch_size=self.args.classifier_bs*2, shuffle=False, drop_last = False)

        self.class_weights = torch.tensor(ds_train.class_weights)



    def validation(self, model, act_fct, epoch):
        # Validation
        model.eval()
        running_loss_v = 0.0
        running_corrects_v = 0

        for batch in tqdm(self.validation_loader):
                inputs = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                cam = compute_cam(inputs, GradCAM, model) 
                plot_inputs = batch['image']


                with torch.set_grad_enabled(False):
                    out = model(inputs)                    
                    loss = F.binary_cross_entropy_with_logits(out, labels, weight=None)
                    running_loss_v += loss.item() * self.args.classifier_bs
                    pred_classes = act_fct(out)>0.5
                    running_corrects_v += torch.sum(pred_classes == labels)


        validation_loss = running_loss_v/self.n_imgs_val
        validation_acc = running_corrects_v/(self.n_imgs_val*OUTPUT_NEURONS)
        wandb.log({"Classifier/Validation Loss": validation_loss})
        wandb.log({"Classifier/Validation Corrects": validation_acc})

        n_val = np.array([N_VAL,self.args.classifier_bs, cam.shape[0]]).min()
        fig, axs = plt.subplots(2,n_val, figsize=(n_val*5, 5))
        for i in range(n_val):
            c = min_max(cam[i,:,:])
            # c = np.zeros((IMG_SIZE)) # TODO

            if(n_val == 1):
                axs[0].imshow(min_max(plot_inputs[i,:,:].numpy().transpose(1,2,0)))
                axs[0].set_title("Contains Virus\nPred: "+str(pred_classes[i].cpu().numpy())+"\nGT: "+str(labels[i].cpu().numpy()), fontsize=8)
                axs[0].set_axis_off()
                axs[1].imshow(c)
                axs[1].set_axis_off()
            else: 
                axs[0,i].imshow(min_max(plot_inputs[i,:,:].numpy().transpose(1,2,0)))
                axs[0,i].set_title("Contains Virus\nPred: "+str(pred_classes[i].cpu().numpy())+"\nGT: "+str(labels[i].cpu().numpy()), fontsize=8)
                axs[0,i].set_axis_off()
                axs[1,i].imshow(c)
                axs[1,i].set_axis_off()


            
        wandb.log({"Classifier/plot": wandb.Image(plt)})
        plt.close(fig)
        print("INFO::Validation Loss = "+str(validation_loss)+" Validation Corrects = "+str(validation_acc))
        return validation_loss
    
    def compute_f1_score(self, dataloader):
        model,_,_ = load_classifier(self.log_path+"/training_state.pth", torch.nn.Sigmoid(), self.args.loss)
        model.to(DEVICE)
        model.eval()
        predictions_lst = []
        labels_lst = []
        for batch in tqdm(dataloader):
                inputs = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                with torch.set_grad_enabled(False):
                    out = model(inputs).cpu().numpy()
                    predictions_lst.extend(out)
                    labels_lst.extend(labels.cpu().int().numpy())
        
        predictions_lst = np.concatenate(np.array(predictions_lst))
        labels_lst = np.concatenate(np.array(labels_lst)).astype(np.bool)

        thresholds = np.arange(0.01,1,0.01)
        f1_lst = []
        precision_lst = []
        recall_lst = []
        for t in thresholds: 
            predictions_thresholded = (predictions_lst >= t)
            true_predictions = (predictions_thresholded == labels_lst)
            tp = np.sum(true_predictions[predictions_thresholded])
            tn = np.sum(true_predictions[~predictions_thresholded])
            false_predictions = (predictions_thresholded != labels_lst)
            fp = np.sum(false_predictions[predictions_thresholded])
            fn = np.sum(false_predictions[~predictions_thresholded])
            precision = tp/(tp+fp+1e-10)
            recall = tp/(tp+fn)
            precision_lst.append(precision)
            recall_lst.append(recall)
            f1_lst.append(2*((precision*recall)/(precision+recall)))
        f1_lst = np.array(f1_lst)
        best_f1 = np.max(f1_lst)
        best_t = thresholds[np.argmax(f1_lst)]            
        return f1_lst, best_f1, best_t

    def evaluation(self, epoch, dataloader):
        act_fct = torch.nn.Sigmoid()

        # Evaluation       
        model,_,_ = load_classifier(self.log_path+"/training_state.pth", torch.nn.Identity(), self.args.loss)
        model.to(DEVICE)
        model.eval()

        running_loss_v = 0.0
        running_corrects_v = 0
        preds = []
        targets = []
        for batch in tqdm(dataloader):
                inputs = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                with torch.set_grad_enabled(False):
                    out = model(inputs)

                    loss = F.binary_cross_entropy_with_logits(out, labels, weight=None)

                    running_loss_v += loss.item() * inputs.size(0)
                    pred_classes = act_fct(out)>0.5

                    running_corrects_v += torch.sum(pred_classes == labels)

                    preds.extend(act_fct(out))
                    targets.extend(labels.int())
                    
        validation_loss = running_loss_v/self.n_imgs_test
        validation_acc = running_corrects_v/(self.n_imgs_test*OUTPUT_NEURONS)

        metrics = compute_classifier_metrics(torch.tensor(preds, device = DEVICE), torch.tensor(targets, device = DEVICE))
                   
        return validation_loss, validation_acc, metrics

    def train_step(self, model, optimizer, scheduler, scaler, epoch):
        model.train()
        running_loss_t = 0.0
        running_corrects_t = 0
        act_fct = torch.nn.Sigmoid()


        for batch in tqdm(self.training_loader):
                inputs = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                with torch.set_grad_enabled(True):       

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.args.classifier_use_amp): 
                       
                        out_full_img = model(inputs)

                        loss = F.binary_cross_entropy_with_logits(out_full_img, labels, weight=None)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    pred_classes = act_fct(out_full_img)>0.5

                    running_corrects_t += torch.sum(pred_classes == labels)
                    running_loss_t += loss.item() * self.args.classifier_bs
        scheduler.step()
        return (running_loss_t/self.n_imgs_train), (running_corrects_t/(self.n_imgs_train*OUTPUT_NEURONS))


    def training(self):
        model = self.init_model(torch.nn.Identity())
        act_fct = torch.nn.Sigmoid()
        optimizer, scheduler = self.init_optim(model)  

        best_val_loss = np.inf
        convergence_epoch = 0
        iteration = 0
        epoch = 0
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.classifier_use_amp)

        while(iteration < self.args.classifier_iters):
        
            # Training
            training_loss, training_acc = self.train_step(model, optimizer, scheduler, scaler, epoch)
            wandb.log({"Classifier/Training Loss": training_loss})
            wandb.log({"Classifier/Training Corrects": training_acc})

            iteration += len(self.training_loader.dataset)
            epoch += 1

            # Validation
            validation_loss = self.validation(model, act_fct, epoch)
            # scheduler.step(validation_loss) 

            # Early stopping
            if(validation_loss<best_val_loss):
                best_val_loss = validation_loss
                convergence_epoch = 0
                # Save model
                dict_saveparams = {}
                set_param(self.training_loader.dataset.path, 'training_paths', dict_saveparams)
                set_param(self.training_loader.dataset.percentage, 'training_data_size', dict_saveparams)
                set_param(self.args.annotation_time, 'annotation_time', dict_saveparams)
                set_param(wandb.run.id, 'wandb_id', dict_saveparams)
                set_param(wandb.run.name, 'wandb_name', dict_saveparams)
                set_param(wandb.run.group, 'wandb_group', dict_saveparams)
                set_param(wandb.run.project, 'wandb_project', dict_saveparams)
                set_param(wandb.run.get_url(), 'wandb_url', dict_saveparams)
                set_param(model.state_dict(), 'model', dict_saveparams)
                set_param(self.args.backbone, 'backbone', dict_saveparams)

                save_dict(dict_saveparams, self.log_path+"/training_state.pth")
            else: 
                convergence_epoch += 1
                if((convergence_epoch >= PATIENCE_EARLY_STOPPING) and (iteration >= MIN_ITERATIONS)):
                    print("INFO::Stop training based on convergence")
                    break

        # Evaluation on valset
        test_loss, test_acc, val_metrics = self.evaluation(epoch, self.validation_loader)
        wandb.log({"Classifier/Val Loss": test_loss})
        wandb.log({"Classifier/Val Acc Bin": test_acc})
        wandb.log({"Classifier/Val accuracy": val_metrics['accuracy']})
        wandb.log({"Classifier/Val specificity": val_metrics['specificity']})
        wandb.log({"Classifier/Val precision": val_metrics['precision']})
        wandb.log({"Classifier/Val recall": val_metrics['recall']})
        wandb.log({"Classifier/Val AUROC": val_metrics['AUROC']})
        wandb.log({"Classifier/Val AP": val_metrics['AP']})
        time.sleep(15)
        out = "Model Val Acc: "+str(test_acc)+"\nModel Val Loss: "+str(test_loss)+"\n\nFurther Metrics:\n"

        for k in val_metrics: 
            out = out+str(k)+": "+str(val_metrics[k])+"\n"
        out = out + "\n"
        
        # Evaluation on testset
        test_loss, test_acc, test_metrics = self.evaluation(epoch, self.test_loader)
        wandb.log({"Classifier/Eval Loss": test_loss})
        wandb.log({"Classifier/Eval Acc Bin": test_acc})
        wandb.log({"Classifier/Eval accuracy": test_metrics['accuracy']})
        wandb.log({"Classifier/Eval specificity": test_metrics['specificity']})
        wandb.log({"Classifier/Eval precision": test_metrics['precision']})
        wandb.log({"Classifier/Eval recall": test_metrics['recall']})
        wandb.log({"Classifier/Eval AUROC": test_metrics['AUROC']})
        wandb.log({"Classifier/Eval AP": test_metrics['AP']})
        time.sleep(15)
        out += "\nModel Test Acc: "+str(test_acc)+"\nModel Test Loss: "+str(test_loss)+"\n\nFurther Metrics:\n"

        for k in test_metrics: 
            out = out+str(k)+": "+str(test_metrics[k])+"\n"
        print(out)

        write_txt(self.log_path+"/Evaluation.txt", out)

        # compute F1 scores on validation set
        val_f1_lst, val_best_f1, val_best_t = self.compute_f1_score(self.validation_loader)
        wandb.log({"Classifier/Val best F1": val_best_f1})
        wandb.log({"Classifier/Val best T": val_best_t})

        plt.close()
        fig = plt.figure()
        plt.plot(np.linspace(0,1,len(val_f1_lst)), val_f1_lst)
        plt.scatter(val_best_t, val_best_f1)
        plt.title("F1 score over multiple thresholds")
        wandb.log({"Classifier/Validation F1": wandb.Image(plt)})
        plt.close(fig)
        time.sleep(15)

        # compute F1 scores on test set
        test_f1_lst, test_best_f1, test_best_t = self.compute_f1_score(self.test_loader)
        wandb.log({"Classifier/Eval best F1": test_best_f1})
        wandb.log({"Classifier/Eval best T": test_best_t})

        # Save final model
        dict_saveparams = {}
        set_param(self.training_loader.dataset.path, 'training_paths', dict_saveparams)
        set_param(self.training_loader.dataset.percentage, 'training_data_size', dict_saveparams)
        set_param(self.args.annotation_time, 'annotation_time', dict_saveparams)
        set_param(wandb.run.id, 'wandb_id', dict_saveparams)
        set_param(wandb.run.name, 'wandb_name', dict_saveparams)
        set_param(wandb.run.group, 'wandb_group', dict_saveparams)
        set_param(wandb.run.project, 'wandb_project', dict_saveparams)
        set_param(wandb.run.get_url(), 'wandb_url', dict_saveparams)
        set_param(model.state_dict(), 'model', dict_saveparams)
        set_param(self.args.backbone, 'backbone', dict_saveparams)
        set_param(val_best_t, 'val_best_t', dict_saveparams)
        set_param(val_best_f1, 'val_best_f1', dict_saveparams)
        set_param(val_f1_lst, 'val_f1_lst', dict_saveparams)
        set_param(val_metrics['AUROC'], 'val_AUROC', dict_saveparams)
        set_param(val_metrics['AP'], 'val_AP', dict_saveparams)
        set_param(test_best_t, 'test_best_t', dict_saveparams)
        set_param(test_best_f1, 'test_best_f1', dict_saveparams)
        set_param(test_f1_lst, 'test_f1_lst', dict_saveparams)
        set_param(test_metrics['AUROC'], 'test_AUROC', dict_saveparams)
        set_param(test_metrics['AP'], 'test_AP', dict_saveparams)

        save_dict(dict_saveparams, self.log_path+"/training_state.pth")

        wandb.run.finish()

        return model, val_best_t, self.training_loader.dataset.path, self.log_path


    def collect_stats(self, stats_loader: DataLoader):
        all_bb_scores = []
        for bb_scores, label, bbox in tqdm(stats_loader):
            if len(bb_scores[0]) > 0:
                for score in bb_scores:
                    all_bb_scores.append(score.item())

        return all_bb_scores
