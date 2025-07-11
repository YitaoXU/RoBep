import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, matthews_corrcoef,
    accuracy_score
)

from bce.loss import CLoss
from bce.utils.constants import DISK_DIR, BASE_DIR
from ..data.data import create_data_loader
from ..model.ReGEP import ReGEP
from bce.model.scheduler import get_scheduler

torch.set_num_threads(12)

class Evaluator:
    def __init__(self, args):
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        self.model = ReGEP.load(args.model_path, device=self.device, strict=False)
        
        self.model.eval()
        
        