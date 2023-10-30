import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "D3G"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.D3G = CN()
_C.MODEL.D3G.NUM_CLIPS = 128
_C.MODEL.D3G.JOINT_SPACE_SIZE = 256

_C.MODEL.D3G.FEATPOOL = CN()
_C.MODEL.D3G.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.D3G.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.D3G.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.D3G.FEAT2D = CN()
_C.MODEL.D3G.FEAT2D.NAME = "pool"
_C.MODEL.D3G.FEAT2D.POOLING_COUNTS = [15, 8, 8, 8]

_C.MODEL.D3G.TEXT_ENCODER = CN()
_C.MODEL.D3G.TEXT_ENCODER.NAME = 'BERT'

_C.MODEL.D3G.PREDICTOR = CN() 
_C.MODEL.D3G.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.D3G.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.D3G.PREDICTOR.NUM_STACK_LAYERS = 8


_C.MODEL.D3G.LOSS = CN()
_C.MODEL.D3G.LOSS.MIN_IOU = 0.3
_C.MODEL.D3G.LOSS.MAX_IOU = 0.7
_C.MODEL.D3G.LOSS.BCE_WEIGHT = 1
_C.MODEL.D3G.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL = 1
_C.MODEL.D3G.LOSS.NEGATIVE_VIDEO_IOU = 0.5
_C.MODEL.D3G.LOSS.SENT_REMOVAL_IOU = 0.5
_C.MODEL.D3G.LOSS.PAIRWISE_SENT_WEIGHT = 0.0
_C.MODEL.D3G.LOSS.CONTRASTIVE_WEIGHT = 0.05
_C.MODEL.D3G.LOSS.TAU_VIDEO = 0.2
_C.MODEL.D3G.LOSS.TAU_SENT = 0.2
_C.MODEL.D3G.LOSS.MARGIN = 0.2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_EPOCH = 1
_C.SOLVER.FREEZE_BERT = 4
_C.SOLVER.ONLY_IOU = 7
_C.SOLVER.SKIP_TEST = 0
# for gaussian weight generation
_C.SOLVER.SIGMA = 0.3 
_C.SOLVER.WINDOW = 4 
_C.SOLVER.TOPK = 10 
_C.SOLVER.THRESH = 0.9
_C.SOLVER.USE_DGA = False

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.5
_C.TEST.CONTRASTIVE_SCORE_POW = 0.5
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
