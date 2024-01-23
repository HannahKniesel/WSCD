# Reproducability and Model weights

We provide the commands for rerunning the main experiments as well as model weights of the best models of our main experiments.

## Table 1: Comparison to State-Of-The-Art

### Supervised, Minimal Labels and Ours
```bash
# Ours(Opt) and Ours(OD)
python Main_Binary.py --project WSCD --seeds 42 123 7353 --annotation_time -1 --data_split train test --dataset herpes
python Main_Binary.py --seeds 42 123 7353 --max_iters 200 --classifier_lr 0.0005 --data_split test train --dataset adeno --lr_t=0.005 --lr_t_final=0.02 --nms_max_iou=0.2 --project WSCD --pseudolabels_use_validation=true  
python Main_Binary.py --seeds 42 123 7353 --max-iters 200 --classifier_lr=0.0005 --data_split test train --dataset noro --lr_t=0.0005 --lr_t_final=0.02 --nms_max_iou=0.01 --project WSCD --pseudolabels_use_validation=true
python Main_Binary.py --seeds 42 123 7353 --max-iters 200 --classifier_lr=0.0005 --data_split test train --dataset papilloma --lr_t=0.001 --lr_t_final=0.02 --nms_max_iou=0.1 --project=WSCD  --pseudolabels_use_validation=true 
python Main_Binary.py --seeds 42 123 7353 --max-iters 200 --classifier_lr=0.0005 --data_split test train --dataset=rota --lr_t=0.001 --lr_t_final=0.02 --nms_max_iou=0.1 --project=WSCD --pseudolabels_use_validation=true

# BB
python Main_BoundingBox.py --project WSCD --seeds 42 123 7353 --annotation_time 38027 --dataset herpes

# Loc 
python Main_Location.py --project WSCD --seeds 42 123 7353 --annotation_time 38027 --dataset herpes
python Main_Location.py --project WSCD --seeds 42 123 7353 --annotation_time 1240 --dataset adeno 
python Main_Location.py --project WSCD --seeds 42 123 7353 --annotation_time 609 --dataset noro
python Main_Location.py --project WSCD --seeds 42 123 7353 --annotation_time 587 --dataset papilloma 
python Main_Location.py --project WSCD --seeds 42 123 7353 --annotation_time 769 --dataset rota


```

### Saliency Based Weakly Supervised Methods

- For computing the saliency maps for GradCAM, and ViT we use pretrained classifiers and then run the `./Comparisons/ComputeCAMs.py` script.

- For TS-CAM we use the [official repo](https://github.com/vasgaowei/TS-CAM), add our own datasets and save the resulting saliency maps. 

- For Reattention we use the [official repo](https://github.com/su-hui-zz/ReAttentionTransformer), add our own datasets and save the resulting saliency maps. 

As these SOTA methods for comparison are based on saliency maps, we use `./Comparisons/Evaluate.py` to compute bounding boxes from the saliency map, include the virus size and evaluate against GT labels. 

### Zero Shot Methods

- For CutLer, we use the [official repo](https://github.com/facebookresearch/CutLER) to generate pseudolabels, which we then use to train a Faster RCNN.

- For SAM, we use the [huggingface implementation](https://huggingface.co/docs/transformers/main/model_doc/sam) to generate pseudolabels, which we then use to train a Faster RCNN. 

Example usage: 
```bash
python Main_BoundingBox.py --project ICLR_SAM_Adeno --dataset adeno --seeds 42 123 7353 --pseudolabels sam --filter_pseudolabels true --size_range 0.6
```

For all compared methods we also include the known virus size. If you are interested in the implementations, please feel free to contact us. 


## Figure 5: Reduced Annotation Time
```bash
# Ours(Opt) and Ours(OD)
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time -1 --data_split train test 
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 28520 --data_split train test 
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 19014 --data_split train test 
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 9507 --data_split train test 
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 3803 --data_split train test 
python Main_Binary.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 1901 --data_split train test 

# Loc
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 38027
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 28520
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 19014 
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 9507
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 3803
python Main_Location.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 1901

# BB
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 38027
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 28520
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 19014 
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 9507
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 3803
python Main_BoundingBox.py --project Reduced_Annotation_Time --seeds 42 123 7353 --annotation_time 1901
```

## Figure 6: Infinite Annotation Time 

```bash
# Ours(Opt) and Ours(OD)
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage -1 --data_split train test 
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.75 --data_split train test 
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.5 --data_split train test 
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.25 --data_split train test 
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.1 --data_split train test 
python Main_Binary.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.05 --data_split train test 

# Loc
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 1
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.75
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.5 
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.25
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.1
python Main_Location.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.05

# BB
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 1
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.75
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.5 
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.25
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.1
python Main_BoundingBox.py --project Infinite_Annotation_Time --seeds 42 123 7353 --percentage 0.05
```
