import sys
sys.path.insert(0, '..')

import argparse
from Variables import *
from Datasets import *
from Transforms import frcnn_transform_resnet101_inference, classification_transform_resnet101_inference
from Utils_Eval import load_classifier
from Detector.utils import collate_fn
from GradCAM import GradCAM, LayerCAM
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from pathlib import Path

transform = classification_transform_resnet101_inference

def generate_cams(dataloader, save_path, description= ""):
    for batch, targets in tqdm(dataloader, desc=description):
        images = [b['image'] for b in batch]
        images = torch.stack(images)
        grayscale_cam = cam(input_tensor=images.to(DEVICE), targets=target_0)
        scores = model(images.to(DEVICE)).detach().cpu()

        # save image, cam, score, gt_boxes
        for img, gray_cam, score, target, b in zip(images, grayscale_cam, scores, targets, batch):
            name = Path(b['path']).parent.stem+"_"+Path(b['path']).stem
            save_as_pickle([img,gray_cam,score,target['boxes'],b['capsideradius']], save_path+"/"+name)


            if(target['boxes'].shape[0]>0):
                fig,axs = plt.subplots(1,2)
                axs[0].imshow(min_max(img.permute(1,2,0)))
                axs[1].imshow(gray_cam.squeeze(), cmap="plasma") 
                plt.title(f"{score.detach().cpu().item():.4f}")
                wandb.log({f"{description} Images": wandb.Image(plt)})
                plt.close()
    return


if __name__ == "__main__":

    print("******************************")
    print("Comparison")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Comparison')

    # General Parameters
    parser.add_argument('--classifier_path', type = str, default="../TrainingRuns/Herpes/Binary/", help='Logging directory (default: ./TrainingRuns/Herpes/Binary/)')
    parser.add_argument('--dataset', type = str, default="herpes", choices=["herpes", "adeno", "noro", "papilloma", "rota"], help='which dataset to use (default:herpes)')
    parser.add_argument('--batch_size', type = int, default=8, help='batch size (default:8)')
    parser.add_argument('--cam', type = str, default="GradCAM", choices = ["GradCAM", "LayerCAM"], help='CAM method (default:GradCAM)')
    parser.add_argument('--backbone', type = str, default="resnet", choices = ["resnet", "vit"], help='architecture (default:resnet)')




    # TODO add more methods
    cam_methods = {"GradCAM": GradCAM, 
                   "LayerCAM": LayerCAM}


    args = parser.parse_args()


    now = datetime.now() # current date and time
    save_path = f"./Comparisons/{args.dataset}/{args.cam}_{args.backbone}/{Path(args.classifier_path).stem}/"
    test_path = save_path+"/test/"
    val_path = save_path+"/val/"
    os.makedirs(test_path, exist_ok = True)
    os.makedirs(val_path, exist_ok = True)

    os.environ['WANDB_PROJECT']= "GenerateComparisons"
    wandb.init(config = args, reinit=True, group = f"{args.cam}_{args.dataset}_{args.backbone}", mode = "online")
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    wandb.run.name = str(date_time)+"_"+str(wandb.run.id)+"_"+str(Path(args.classifier_path).stem)

   
    # define datasets
    if(args.dataset == "herpes"):
        ds_test = HerpesLabelGeneration_Dataset("."+HERPES_TEST_DATA_PATH, transform, 42, -1, -1, CLASSIFICATION_TIMINGS, data_paths=[], preload= False) #, num_imgs=self.num_data, num_virus=self.args.num_virus, start_idx = self.args.start_idx)
        ds_val = HerpesLabelGeneration_Dataset("."+HERPES_VAL_DATA_PATH, transform, 42, -1, -1, CLASSIFICATION_TIMINGS, data_paths=[], preload= False) #, num_imgs=self.num_data, num_virus=self.args.num_virus, start_idx = self.args.start_idx)
    elif(args.dataset == "adeno"): 
        ds_test = TEMLabelGeneration_Dataset("."+ADENO_TEST_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
        ds_val = TEMLabelGeneration_Dataset("."+ADENO_VAL_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
    elif(args.dataset == "noro"):
        ds_test = TEMLabelGeneration_Dataset("."+NORO_TEST_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
        ds_val = TEMLabelGeneration_Dataset("."+NORO_VAL_DATA_PATH, transform, 42, -1, data_paths=[], preload= False)     
    elif(args.dataset == "papilloma"): 
        ds_test = TEMLabelGeneration_Dataset("."+PAP_TEST_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
        ds_val = TEMLabelGeneration_Dataset("."+PAP_VAL_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
    elif(args.dataset == "rota"): 
        ds_test = TEMLabelGeneration_Dataset("."+ROT_TEST_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
        ds_val = TEMLabelGeneration_Dataset("."+ROT_VAL_DATA_PATH, transform, 42, -1, data_paths=[], preload= False) 
    data_loader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)


    # load model
    model,_,_,_ = load_classifier(args.classifier_path+"/training_state.pth")
    model.to(DEVICE)
    model.eval()   
    for param in model.parameters():
        param.requires_grad = True
    target_category_0 = np.zeros(1,).astype(int)
    target_0 = [ClassifierOutputTarget(category) for category in target_category_0] *args.batch_size

    if(args.backbone == "resnet"):
        target_layer = [model[0].model.layer4[-1]]
        cam = cam_methods[args.cam](model=model, target_layers=target_layer, use_cuda=False) #, reshape_transform=reshape_transform) # should still computed on GPU, since model and target already are on GPU
    elif(args.backbone == "vit"):
        target_layer = [model[0].model.encoder.layers.encoder_layer_11.ln_1] # vit
        cam = cam_methods[args.cam](model=model, target_layers=target_layer, use_cuda=False, reshape_transform=reshape_transform) # should still computed on GPU, since model and target already are on GPU
    else: 
        print(f"ERROR::No such backbone {args.backbone}")

    # compute CAMs on the validation set
    generate_cams(data_loader_val, val_path, "validation")
    print(f"Saved {args.cam} on validation set to: {val_path+'/'}")
    

    # compute CAMs on the test set
    generate_cams(data_loader_test, test_path, "test")
    print(f"Saved {args.cam} on test set to: {test_path+'/'}")


