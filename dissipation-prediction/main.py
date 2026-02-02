from args import get_args

import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import load_data, prepare_ml_data
from src.model_runner import run_model
from src.plots import tsf_comparison, plot_metrics


# Parse arguments
args = get_args()

DT = args.DT 
m = int(args.lookback / DT)
h = int(args.tau / DT)
label_len = int(args.label / DT)

print("\n # =================================== # ")
print("             --- Device Info ---           ")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device} ")
print(" # =================================== # ")


print("\n # ============================================= # ")
print(f"        [INFO] Selected input: {args.inp}         ")
print(" # ============================================= # ")

X, y, label = load_data(args)


datasets, datasets_norm, train_loader, val_loader, test_loader, normalizer = prepare_ml_data(X, y, m, h, label_len, args)

#X_train, y_train = datasets["train"]
#X_val,   y_val   = datasets["val"]
X_test,  y_test  = datasets["test"]

X_train_norm, y_train_norm = datasets_norm["train"]
#X_val_norm,   y_val_norm   = datasets_norm["val"]
X_test_norm,  y_test_norm  = datasets_norm["test"]


print("\n # ========================================== # ")
print(f"               [INFO] Run the model.           ")
print(" # ========================================== # ")

preds, trues = run_model(
    args=args,
    no_inputs=X.shape[-1],
    y_train_norm=y_train_norm,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    normalizer=normalizer,
    device=device,
    DT=DT,
    m=m,
    h=h,
    label_len=label_len
)


print("\n # ============================================= # ")
print(f"    [INFO] You have successfully run the code.  ")
print(" # ============================================ # ")
