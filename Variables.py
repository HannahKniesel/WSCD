import torch

IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = ['naked', 'budding', 'enveloped']

PATIENCE_EARLY_STOPPING = 5
MIN_ITERATIONS = 230000
N_VAL = 3


ADENO_TRAIN_DATA_PATH = "./Data/LargeScaleTEM/Adenovirus/train/Data/"
ADENO_TEST_DATA_PATH = "./Data/LargeScaleTEM/Adenovirus/test/Data/"
ADENO_VAL_DATA_PATH = "./Data/LargeScaleTEM/Adenovirus/validation/Data/"

NORO_TRAIN_DATA_PATH = "./Data/LargeScaleTEM/Norovirus/train/Data/"
NORO_TEST_DATA_PATH = "./Data/LargeScaleTEM/Norovirus/test/Data/"
NORO_VAL_DATA_PATH = "./Data/LargeScaleTEM/Norovirus/validation/Data/"

PAP_TRAIN_DATA_PATH = "./Data/LargeScaleTEM/Papilloma/train/Data/"
PAP_TEST_DATA_PATH = "./Data/LargeScaleTEM/Papilloma/test/Data/"
PAP_VAL_DATA_PATH = "./Data/LargeScaleTEM/Papilloma/validation/Data/"

ROT_TRAIN_DATA_PATH = "./Data/LargeScaleTEM/Rotavirus/train/Data/"
ROT_TEST_DATA_PATH = "./Data/LargeScaleTEM/Rotavirus/test/Data/"
ROT_VAL_DATA_PATH = "./Data/LargeScaleTEM/Rotavirus/validation/Data/"

HERPES_TEST_DATA_PATH = "./Data/Herpes/Crops/Test/Data/"
HERPES_VAL_DATA_PATH = "./Data/Herpes/Crops/Val/Data/"
HERPES_TRAIN_DATA_PATH = "./Data/Herpes/Crops/Train/Data/"

CLASSIFICATION = "bin"
LOCATION = "loc"
BOUNDINGBOX = "bb"

CLASSIFICATION_TIMINGS = "./Data/Herpes/Crops/TimingsBinary.pkl"
LOCATION_TIMINGS = "./Data/Herpes/Crops/TimingsLocation.pkl"
BOUNDINGBOX_TIMINGS = "./Data/Herpes/Crops/TimingsBoundingBox.pkl"

EM_PRETRAINED_WEIGHTS = "./pretrained_models/cem500k_mocov2_resnet50_200ep_pth.tar"

MAX_NUM_OBJ_HERPES = 10
HERPES_CAPSIDE_SIZE = 165 
MAX_IOU = 0.01


OUTPUT_NEURONS = 1


BATCH_SIZE = 1# 32
