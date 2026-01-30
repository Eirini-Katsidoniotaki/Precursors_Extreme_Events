from args import get_args

import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from src.data_utils import load_ftle, load_fourier, split_train_val_test
from src.data_normalization import DataNormalizer
from src.create_sequences import SequenceCreator
from src.model_runner import run_model

# Parse arguments
args = get_args()
DT = args.DT_base * args.downscale_factor


print(" # ============================================= # ")
print(f"        [INFO] Selected input: {args.inp}         ")
print(" # ============================================= # ")

if args.inp == "FTLE":
    X, y, label = load_ftle(args)
elif args.inp == "Fourier":
    X, y, label = load_fourier(args)


print("\n # ============================================ # ")
print(" [INFO] Split data into train - validation - test:")
print(" # ============================================ # ")

X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, args)


print("\n # ============================================ # ")
print("           [INFO] Normalize datasets:               ")
print(" # ============================================ # ")

normalizer = DataNormalizer()
normalizer.fit(X_train, y_train)

datasets = {
    "train": (X_train, y_train),
    "val":   (X_val, y_val),
    "test":  (X_test, y_test),
}

normalized = {}

for name, (X, y) in datasets.items():
    print(f"[INFO] Normalizing {name} data...")
    normalized[name] = normalizer.transform(X, y)

X_train_norm, y_train_norm = normalized["train"]
X_val_norm, y_val_norm     = normalized["val"]
X_test_norm, y_test_norm   = normalized["test"]

print("[INFO] Normalization statistics:")
print(normalizer.stats())

# Inverse normalization (e.g. predictions back to original scale)
#_, y_rec = normalizer.inverse_transform(yn=yn)


print("\n # ================================ # ")
print("   [INFO] Create sequential datasets.   ")
print(" # ================================ # ")


m = int(args.lookback / DT)
h = int(args.tau / DT)
label_len = int(args.label / DT)

print(f"\n [INFO] Lookback window = {args.lookback} seconds, which translates to {m} time steps.")
print(f"\n [INFO] Prediction horizon tau = {args.tau} seconds, which translates to {h} time steps.")
print(f"\n [INFO] Label = {args.label} seconds, which translates to {label_len} time steps.")

seq_gen = SequenceCreator(m=m, h=h)

X_train_seq, y_train_seq = seq_gen.create_sequences(X_train_norm, y_train_norm, label_len, to_torch=True) 
X_val_seq,   y_val_seq   = seq_gen.create_sequences(X_val_norm,   y_val_norm,   label_len, to_torch=True) 
X_test_seq,  y_test_seq  = seq_gen.create_sequences(X_test_norm,  y_test_norm,  label_len, to_torch=True) 

print("\n--- Dataset shapes after sequence creation ---")
print(f"Train: X {X_train_seq.shape}, y {y_train_seq.shape}")
print(f"Val  : X {X_val_seq.shape},   y {y_val_seq.shape}")
print(f"Test : X {X_test_seq.shape},  y {y_test_seq.shape}")
print("===================================\n")


print("\n # ==================================== # ")
print("   [INFO] Create batched datasets for ML.   ")
print(" # ====================================== # ")

datasets_seq = {
    "train": (X_train_seq, y_train_seq),
    "val":   (X_val_seq,   y_val_seq),
    "test":  (X_test_seq,  y_test_seq),
}

loaders = {}

for split, (X, y) in datasets_seq.items():
    shuffle = (split == "train")
    dataset = TensorDataset(X, y)
    loaders[split] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )

train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]



print("\n =================================== ")
print("\n --- Device Info --- ")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device} ")
print(" =================================== ")



print(" # ========================================== # ")
print(f"               [INFO] Run the model.           ")
print(" # ========================================== # \n")

preds, trues = run_model(
    args=args,
    X=X,
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


print(" # ========================================== # ")
print(f"       [INFO] Evaluate model's accuracy.       ")
print(" # ========================================== # \n")
