import numpy as np
import glob
from PIL import Image
import gzip, pickle, pickletools
import torch
import random
from GradCAM.utils.model_targets import ClassifierOutputTarget
import tifffile

from torchmetrics import Accuracy, Specificity, Precision, Recall, AUROC, AveragePrecision

from Variables import *

mse_loss = torch.nn.MSELoss()

def deterministic(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)
    print("INFO::Deterministic true with seed="+str(seed))
    return

def compute_classifier_metrics(pred, target):
    # mean(pred == target)
    acc = Accuracy(task = 'binary', threshold = 0.5).to(DEVICE)
    acc = acc(pred, target)

    # TN/(TN+FP)
    specificity = Specificity(task = 'binary', threshold = 0.5).to(DEVICE)
    specificity = specificity(pred, target)

    # TP/(TP+FP) What proportion of positive identifications was actually correct? --> Are there many FP detections?
    precision = Precision(task = 'binary', threshold = 0.5).to(DEVICE)
    precision = precision(pred, target)

    # TP/(TP+FN) What proportion of actual positives was identified correctly? --> have all TP been detected?
    recall = Recall(task = 'binary', threshold = 0.5).to(DEVICE)
    recall = recall(pred, target)

    # summarizes ROC curve
    auroc = AUROC(task = 'binary').to(DEVICE)
    auroc = auroc(pred, target)

    # summarized Precision-Recall curve
    ap = AveragePrecision(task = 'binary').to(DEVICE)
    ap = ap(pred, target)

    return {'accuracy': acc, 'specificity': specificity, 'precision': precision, 'recall':recall, 'AUROC': auroc, 'AP': ap}

def compute_capside_size(pixelsize_in_m, capside_size_in_nm = HERPES_CAPSIDE_SIZE):
    pixelsize_in_nm = pixelsize_in_m * 10**9
    capside_size_in_px = capside_size_in_nm/pixelsize_in_nm
    return round(capside_size_in_px)


def perfectGradCAM(locations, capsidradius): # center=None, sig = 1):
    sig = capsidradius/IMG_SIZE[0]
    cams = []
    for loc in locations: 
        if(loc[0]<0):
            continue
        center = [int(loc[0]), int(loc[1])]
        Y, X = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        max_dist = np.sqrt(IMG_SIZE[0]**2 + IMG_SIZE[1]**2)/2
        dist_from_center = dist_from_center/max_dist
        mask = gaussian(dist_from_center, mu = 0, sig = sig)
        cams.append(mask)
    if(len(cams) == 0):
        cams = np.zeros((1,IMG_SIZE[0], IMG_SIZE[1]))
    else:
        cams = np.max(np.stack(cams),axis = 0)[None,:,:]
    return cams


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def compute_cam(input_tensor, mechanism, model, layer_idx = -1):
    transform = None
    try:
        bs = input_tensor.shape[0]
    except: 
        bs = input_tensor[0].shape[0]
    try:
        target_layer = [model[0].layer4[layer_idx]]
    except: 
        
        try:
            target_layer = [model[0].model.layer4[layer_idx]]
        except:
            target_layer = [model[0].model.encoder.layers.encoder_layer_11.ln_1] # vit
            transform = reshape_transform
    target_category_0 = np.zeros(1,).astype(int)
    target_0 = [ClassifierOutputTarget(category) for category in target_category_0] *bs
    cam = mechanism(model=model.cpu(), target_layers=target_layer, use_cuda=False, reshape_transform=transform) # should still computed on GPU, since model and target already are on GPU
    
    grayscale_cam = cam(input_tensor=input_tensor.cpu(), targets=target_0)

    model = model.cuda()
    input_tensor = input_tensor.cuda()
    return grayscale_cam

def crisp_mask(mask_torch):
    return torch.sigmoid(100000*(mask_torch-0.9999))   



def mask_input(transformed_mask, input_img, masking, bg_dataset, norm_transform):
    if(transformed_mask.is_cuda):
        input_img = input_img.to("cuda")
    # mask input 
    if(masking == "inpainting"):
        bg = 0 #np.random.randint(0, len(bg_dataset))
        bg = bg_dataset[bg][0]['image'].unsqueeze(0)
        if(transformed_mask.is_cuda):
            bg = bg.to("cuda")
        
        model_in = torch.mul(transformed_mask,input_img) + torch.mul((1-transformed_mask),bg)
        model_in = norm_transform(model_in)

    elif(masking == "mean"):
        input_img = norm_transform(input_img) # if applied before, masking should happen with mean of pretrained ds, since 0 is mean based on z-score normalization
        model_in = torch.mul(transformed_mask,input_img)
    elif(masking == "zeros"):
        model_in = torch.mul(transformed_mask,input_img)
        model_in = norm_transform(model_in) # if applied after, masking should happen with 0

    return model_in

def generate_masks_from_positions(positions, capside_radius):
    masks = []
    for pos in positions:
        mask, _ = create_circular_mask(IMG_SIZE[0],IMG_SIZE[1], center=pos, radius = capside_radius)
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)[:,None,:,:]).float()

def generate_masks_from_boxes(boxes):
    masks = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        x = xmin + ((xmax-xmin)/2)
        y = ymin + ((ymax-ymin)/2)
        mask = create_elipsodial_mask(IMG_SIZE[0],IMG_SIZE[1], center=[x,y], radius = ((xmax-xmin)/2, (ymax-ymin)/2))
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)[:,None,:,:]).float()

def generate_BBmasks_from_positions(positions, capside_radius):
    masks = []
    for pos in positions:
        mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
        x,y = pos
        mask[int(np.max([x-capside_radius, 0])):int(np.min([x+capside_radius, IMG_SIZE[0]])), int(np.max([y-capside_radius, 0])):int(np.min([y+capside_radius, IMG_SIZE[1]]))] = 1
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)[:,None,:,:]).float()

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask, dist_from_center-radius

def create_elipsodial_mask(h, w, center=None, radius=None):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = (((X - center[0])**2)/(radius[0]**2) + ((Y-center[1])**2)/(radius[1]**2)) #np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= 1 #IMG_SIZE[0]/2
    return mask



def add_blur_mask_torch(mask, dist_from_center, sig):
    h = mask.shape[-2]
    w = mask.shape[-1]
    max_dist = torch.sqrt(torch.tensor(h**2 + w**2))/2
    init_shape = mask.shape
    mask = mask.view(-1)
    dist_from_center = dist_from_center.to(DEVICE)
    dist_from_center = dist_from_center.repeat(init_shape[0],1,1,1)   
    dist_from_center = dist_from_center.view(-1)
    dist_from_center = dist_from_center/max_dist
    dist_from_center[mask.bool()] = 0
    mask = mask.view(init_shape)
    dist_from_center = dist_from_center.view(init_shape)
    mask = gaussian_torch(dist_from_center, mu = 0, sig = sig)
    return mask

def gaussian_torch(x, mu, sig):
    return torch.exp(-torch.pow(torch.tensor(x - mu), torch.tensor(2.)) / (2 * torch.pow(torch.tensor(sig), torch.tensor(2.))))

def gaussian_gradient_torch(x, mu, sig):
    return torch.abs(-1*((x-mu)/sig**2)*gaussian_torch(x,mu,sig))

def add_blur_mask(mask, dist_from_center, sig, radius):
    h = mask.shape[-2]
    w = mask.shape[-1]
    max_dist = np.sqrt(h**2 + w**2)/2
    init_shape = mask.shape
    mask = mask.reshape(-1)
    dist_from_center = dist_from_center.reshape(-1)
    dist_from_center = np.maximum(dist_from_center, 0)  
    dist_from_center = dist_from_center/max_dist
    mask = mask.reshape(init_shape)
    dist_from_center = dist_from_center.reshape(init_shape)
    mask = gaussian(dist_from_center, mu = 0, sig = sig)
    return mask

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_gradient(x, mu, sig):
    return np.abs(-1*((x-mu)/sig**2)*gaussian(x,mu,sig))

def create_blur_circular_mask(h, w, center=None, radius=None, sig = 1):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    max_dist = np.sqrt(h**2 + w**2)/2

    mask = dist_from_center <= radius
    dist_from_center = dist_from_center - radius
    init_shape = mask.shape
    mask = mask.reshape(-1)
    dist_from_center = dist_from_center.reshape(-1)
    dist_from_center = dist_from_center/max_dist
    dist_from_center[mask.astype(bool)] = 0
    mask = mask.reshape(init_shape)
    dist_from_center = dist_from_center.reshape(init_shape)
    mask = gaussian(dist_from_center, mu = 0, sig = sig)
    return mask

# converts a 3 channel rgb image to 1 channel grayscale image
def rgb_to_gray(img):
    return img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722

def open_image(path):
    img = Image.open(path)
    img = np.array(img)
    try: 
        img = rgb_to_gray(img)
    except: 
        pass
    img = img.squeeze()
    return img

def open_tif_with_properties(path):
    with tifffile.TiffFile(path) as tif:
        properties = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            properties[name] = value
        image = tif.pages[0].asarray()
    try:
        magnification = properties['OlympusSIS']['magnification']
        pixelsize = properties['OlympusSIS']['pixelsizex']
        properties = {'magnification': magnification, 'pixelsize': pixelsize, 'path': path}
    except:
        print("ERROR:: properties of file: "+str(path))
        print(properties)
    return image, properties

def min_max_torch(volume):
    if(torch.max(volume) == torch.min(volume)):
        if(torch.max(volume)> 1):
            return torch.ones_like(volume)
        elif(torch.min(volume)<0):
            return torch.zeros_like(volume)
        else: 
            return volume
    return (volume - torch.min(volume))/(torch.max(volume)- torch.min(volume))

def min_max_np(volume):
    if(np.max(volume) == np.min(volume)):
        if(np.max(volume)> 1):
            return np.ones_like(volume)
        elif(np.min(volume)<0):
            return np.zeros_like(volume)
        else: 
            return volume
    return (volume - np.min(volume))/(np.max(volume)- np.min(volume))

def min_max(volume):
    try: 
        vol = min_max_np(volume)
    except: 
        vol = min_max_torch(volume)
    return vol 

# reads pickled data
def read_pickle(path):
    with gzip.open(path, 'rb') as f:
        p = pickle.Unpickler(f)
        data = p.load()
    return data

#saves list of values into pkl file
def save_as_pickle(lst, path):
    with gzip.open(str(path+".pkl"), 'wb') as f:
        pickled = pickle.dumps(lst)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)
        #pickle.dump(lst, f)
    if(type(lst) is list):
        lst.clear()
    return True


# Utils to save and load model parameters
def load_param(default_val, name, ckpt):
    try:
        variable = ckpt[name]
    except: 
        print("Did not load variable "+str(name)+" from checkpoint.")
        variable = default_val
    return variable

def save_dict(dict_val, path):
    try:
        torch.save(dict_val, path)
        return True
    except: 
        return False

def set_param(val, name, dict_save):
    try:
        dict_save[name] = val
    except: 
        print("WARNING:: Did not save parameter: "+str(name))
    return

def write_txt(path, txt):
    f = open(path, "w")
    f.write(txt)
    f.close()