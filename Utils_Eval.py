import torch 
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

from Classifier.Models import Classifier_ResNet50, Classifier_ResNet101, Classifier_Oracle, Classifier_ViT


from Variables import *

def get_classifier_wandb_params(path):
    checkpoint = torch.load(path)
    c_path= path
    c_project=checkpoint['wandb_project']
    c_group=checkpoint['wandb_group']
    c_url=checkpoint['wandb_url']
    return c_path, c_project, c_group, c_url


def load_classifier(path, final_act = torch.nn.Sigmoid(), loss = None):
    # load parameters
    checkpoint = torch.load(path)
    backbone = checkpoint['backbone']

    try:
        best_t = checkpoint['best_t']  
    except: 
        best_t = -1  
    data_paths = checkpoint['training_paths']

    # load model
    if(loss == "oracle"):
        model = Classifier_Oracle()
    else: 
        if(backbone == "resnet50"):
            model = Classifier_ResNet50()
        elif(backbone == "resnet101"):
            model = Classifier_ResNet101()
        elif(backbone == "vit"):
            model = Classifier_ViT()    
        model = torch.nn.Sequential(model, final_act)  
        model.load_state_dict(checkpoint['model'], strict=True)  
    
    return model, best_t, data_paths


def iou(cam, mask):
    cam = cam.squeeze().reshape(-1)
    mask = mask.squeeze().reshape(-1)
    cam = (cam).bool()
    mask = mask.bool()

    tp = cam & mask
    fp = cam & (~mask)
    fn = mask & (~cam)

    tp_val = torch.sum(tp)
    fp_val = torch.sum(fp)
    fn_val = torch.sum(fn)

    # both masks do not contain virus
    if((tp_val+fp_val+fn_val) == 0): 
        return -1
    
    # precision = tp_val/(tp_val+fp_val)
    # recall = tp_val/(tp_val+fn_val)
    jaccard = tp_val/(tp_val+fp_val+fn_val)
    # dice = (2*tp_val)/(2*tp_val+fp_val+fn_val)
    return jaccard


def compute_iou2(mask, cam, threshold):
    iou_metric = JaccardIndex(num_classes = 2, average=None, ignore_index = 0) 
    cam = (cam>threshold).astype(bool)
    mask = (mask>0.9).astype(bool)
    cam = np.stack([cam, ~cam])
    mask = np.stack([mask, ~mask])
    cam = torch.from_numpy((cam))
    mask = torch.from_numpy((mask))
    return iou_metric(cam.bool(), mask.bool())

def compute_iou(mask, cam, threshold):
    cam = np.array(cam)
    """if(len(cam.shape)>3):
        cam = cam.squeeze()
        cam = np.max(cam, axis = 0)
    print("Pred Mask Shape"+str(cam.shape))"""
    cam = (cam>threshold).astype(bool)
    mask = (mask>0.9).astype(bool)
    cam = torch.from_numpy(cam)
    mask = torch.from_numpy(mask)
    return iou(cam, mask)

def compute_ious(masks, cams, threshold, predictions = [], compute_on_empty_mask = True, n_virus = []):
    iou_lst_empties = []
    iou_lst = []

    if(len(n_virus)):
        n_virus_unique = np.unique(n_virus)
        iou_lst = [[] for virus in n_virus_unique]

    for i, (mask,cam) in enumerate(zip(masks,cams)):
        # don't compute iou on empty mask, when variable is set
        if((not compute_on_empty_mask) and (np.sum(mask) == 0)):
            continue

        # use predictions for CAM generation
        if(len(predictions) > 0):
            pred = predictions[i]
            if(pred<0.5):
                cam = np.zeros_like(cam)
        
        iou_val = float(compute_iou(mask, cam, threshold))

        if(len(n_virus)):
            curr_n_virus = int(n_virus[i])
            iou_lst[curr_n_virus].append(iou_val)
        else:
            if(np.sum(mask) == 0):
                iou_lst_empties.append(iou_val)
            else:
                iou_lst.append(iou_val) 

    if(len(n_virus)):
        return_lst = [np.mean(lst) for lst in iou_lst]
        flat_list = [item for sublist in iou_lst for item in sublist]
        flat_list2 = [item for sublist in iou_lst[1:] for item in sublist]
        averages = [np.mean(flat_list), np.mean(flat_list2)]
        return return_lst, averages
        
    else:
        if(compute_on_empty_mask):
            return np.mean(iou_lst), np.mean(iou_lst_empties)
        else: 
            return np.mean(iou_lst)

def get_best_threshold(masks, cams, predictions = []):
    best_iou = 0
    best_t = 0
    thresholds = np.arange(0.05, 1.00, 0.05)
    for threshold in thresholds: 
        ious = compute_ious(masks, cams, threshold, predictions = predictions, compute_on_empty_mask = False)
        # assumes the same number of patches with no virus as patches with virus
        # print("Current threshold: "+str(threshold)+" with IOU on non empties: "+str(ious))
        if(ious>best_iou):
            best_t = threshold
            best_iou = ious
    return best_iou, best_t

def compute_iouBACKUP(masks, cams, threshold, predictions = None):
    iou_metric = JaccardIndex(num_classes = 2, average=None, ignore_index = 0) 
    iou = []
    for i, (mask,cam) in enumerate(zip(masks,cams)):
        if(bool(predictions)):
            pred = predictions[i]
            if(pred>0):
                cam = (cam>threshold).astype(bool)
                mask = (mask>0.9).astype(bool)
                cam = np.stack([cam, ~cam])
                mask = np.stack([mask, ~mask])
                cam = torch.from_numpy((cam))
                mask = torch.from_numpy((mask))
                iou.append(iou_metric(cam, mask))
        else: 
            cam = (cam>threshold).astype(bool)
            mask = (mask>0.9).astype(bool)
            cam = np.stack([cam, ~cam])
            mask = np.stack([mask, ~mask])
            cam = torch.from_numpy((cam))
            mask = torch.from_numpy((mask))
            iou.append(iou_metric(cam, mask))
    return np.mean(iou)






def compute_matrix(bb_mask_pred, bb_mask):
    attention = (bb_mask_pred).astype(bool) # thresholded attention map
    tp = attention.astype(bool) & bb_mask.astype(bool)
    tn = (~attention.astype(bool)) & (~bb_mask.astype(bool))
    fp = attention.astype(bool) & (~bb_mask.astype(bool))
    fn = bb_mask.astype(bool) & (~attention.astype(bool))

    tp_val = np.sum(tp.astype(np.int16)) #, axis = (1,2))
    fp_val = np.sum(fp.astype(np.int16)) #, axis = (1,2))
    fn_val = np.sum(fn.astype(np.int16)) #, axis = (1,2))
    tn_val = np.sum(tn.astype(np.int16)) #, axis = (1,2))

    return tp_val, fp_val, fn_val, tn_val
    
    

def compute_score(bb_mask_pred, bb_mask, save_to = "", idx = 0):
    attention = (bb_mask_pred).astype(bool) # thresholded attention map
    tp = attention.astype(bool) & bb_mask.astype(bool)
    fp = attention.astype(bool) & (~bb_mask.astype(bool))
    fn = bb_mask.astype(bool) & (~attention.astype(bool))

    tp_val = np.sum(tp.astype(np.int16), axis = (1,2))
    fp_val = np.sum(fp.astype(np.int16), axis = (1,2))
    fn_val = np.sum(fn.astype(np.int16), axis = (1,2))
    
    precision = tp_val/(tp_val+fp_val+1e-10)
    recall = tp_val/(tp_val+fn_val+1e-10)
    jaccard = tp_val/(tp_val+fp_val+fn_val+1e-10)
    dice = (2*tp_val)/(2*tp_val+fp_val+fn_val+1e-10)
    
    if(save_to):
        for i in range(tp_val.shape[0]):
            fig, axs = plt.subplots(1,5, figsize = (15,5))
            plt.tight_layout()
            axs[0].set_title("GT")
            axs[0].imshow(bb_mask[i], vmax = 1, vmin = 0)
            axs[1].set_title("Predicted IOU: "+str(jaccard[i]))
            axs[1].imshow(attention[i], vmax = 1, vmin = 0)
            axs[2].set_title("TP "+str(tp_val[i]))
            axs[2].imshow(tp[i], vmax = 1, vmin = 0)
            axs[3].set_title("FP "+str(fp_val[i]))
            axs[3].imshow(fp[i], vmax = 1, vmin = 0)
            axs[4].set_title("FN "+str(fn_val[i]))
            axs[4].imshow(fn[i], vmax = 1, vmin = 0)
            plt.savefig(save_to+str(idx)+str(i)+"_TP-FP-FN.png")
            plt.close()

    precision = np.mean(precision)
    recall = np.mean(recall)
    jaccard = np.mean(jaccard)
    dice = np.mean(dice)


    return precision, recall, jaccard, dice
    

def visualize_prediction(img, prediction, label, save_to, idx):
    plt.figure()
    plt.imshow(img)
    plt.title("Label: "+str(label)+"\nPrediction: "+str(prediction))
    plt.savefig(save_to+str(idx)+"_Prediction.png")
    plt.close()

def visualize_prediction_detection(img, mask_predicted, mask_true, save_to, idx):
    """mask_predicted = np.zeros_like(img) + 0.3
    mask_true = np.zeros_like(img) + 0.3"""

    mask_predicted[mask_predicted == 0] = 0.3
    mask_true[mask_true == 0] = 0.3

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(mask_true.transpose(1,0)*img)
    ax[1].imshow(mask_predicted.transpose(1,0)*img)
    ax[0].set_title("Label")
    ax[1].set_title("Prediction")
    plt.savefig(save_to+str(idx)+"_Prediction.png")
    # plt.show()
    plt.close()
    return



def plot_best_results(results_arr, thresholds, attention_names, title, save_to, minimum = False):
    if(minimum):
        best = np.min(results_arr, axis = -1)
        best_idx = np.argmin(results_arr, axis = -1)
    else: 
        best = np.max(results_arr, axis = -1)
        best_idx = np.argmax(results_arr, axis = -1)
    thresholds = np.array(thresholds)
    threshold = thresholds[best_idx.astype(np.int32)]

    titles = []
    for i in range(len(attention_names)):
        titles.append(attention_names[i]+"\n"+str(threshold[i]))

    plt.figure()
    fig = plt.figure(figsize = ((len(attention_names)/2)*5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(titles,best)
    ax.set_xlabel("Attention")
    ax.set_ylabel("Performance")
    plt.title(title)
    plt.savefig(save_to+"/_"+title+"_bar.jpg", bbox_inches='tight')
    plt.close()
    return threshold


def plot_results(results_arr, thresholds, attention_names, title, save_to):
    titles = []
    for i in range(len(attention_names)):
        titles.append(attention_names[i]+"\n"+str(thresholds[i]))

    plt.figure()
    fig = plt.figure(figsize = ((len(attention_names)/2)*5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(titles,results_arr)
    ax.set_xlabel("Attention")
    ax.set_ylabel("Performance")
    plt.title(title)
    plt.savefig(save_to+"/_"+title+"_bar.jpg", bbox_inches='tight')
    plt.close()
    return 