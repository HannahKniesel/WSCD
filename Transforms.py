import sys
sys.path.insert(0,'..')
import torchvision.transforms
from Detector import transforms as T

# Faster RCNN transforms
frcnn_transform_resnet101_training = []
frcnn_transform_resnet101_training.append(T.RandomHorizontalFlip(0.5))
frcnn_transform_resnet101_training.append(T.RandomVerticalFlip(0.5))
frcnn_transform_resnet101_training.append(T.Resize(224))
frcnn_transform_resnet101_training.append(T.ToRGB())
frcnn_transform_resnet101_training = T.Compose(frcnn_transform_resnet101_training)

frcnn_transform_resnet101_inference = []
frcnn_transform_resnet101_inference.append(T.Resize(224))
frcnn_transform_resnet101_inference.append(T.ToRGB())
frcnn_transform_resnet101_inference = T.Compose(frcnn_transform_resnet101_inference)

frcnn_transform_resnet50_training = []
frcnn_transform_resnet50_training.append(T.RandomHorizontalFlip(0.5))
frcnn_transform_resnet50_training.append(T.RandomVerticalFlip(0.5))
frcnn_transform_resnet50_training.append(T.Resize(224))
frcnn_transform_resnet50_training = T.Compose(frcnn_transform_resnet50_training)

frcnn_transform_resnet50_inference = []
frcnn_transform_resnet50_inference.append(T.Resize(224))
frcnn_transform_resnet50_inference = T.Compose(frcnn_transform_resnet50_inference)


# Classification transforms
classification_transform_resnet101_training = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(degrees=(0, 180)),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # to RGB for imagenet pretrained weights
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
                
classification_transform_resnet101_inference = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # to RGB for imagenet pretrained weights
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

# resnet 50 uses EM pretrained weights, hence add normalization
classification_transform_resnet50_inference =  torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.Normalize((0.58331613), (0.09966064))])

classification_transform_resnet50_training =  torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.Normalize((0.58331613), (0.09966064)),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip()])

# pseudolabels 
pseudolabels_transform_resnet101 = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # to RGB for imagenet pretrained weights
                ]) 
norm_resnet101 = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

pseudolabels_transform_resnet50 =  torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                ])
norm_resnet50 = torchvision.transforms.Compose([torchvision.transforms.Normalize((0.58331613), (0.09966064))])


masking_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # to RGB for imagenet pretrained weights
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]) 



