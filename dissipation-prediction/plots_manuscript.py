import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

from src.plots import tsf_comparison, plot_metrics



print("\n# =================================================== #")
print("   [INFO] Plots for the Results section in the manuscript")
print("# ===================================================== #")

# ============================================================
# (1) Define where results are stored on disk
# ------------------------------------------------------------
# Each experiment saves:
#   - pred.npy  : model predictions
#   - true.npy  : ground-truth targets
# inside a folder like:
#   outputs/tau_<tau>/<input_type>/results/
#
# We build a dictionary that maps (input_type, tau) -> results_path
# so downstream plotting utilities can easily locate files.
# ============================================================
taus = [2, 5, 7, 10, 12, 15]          # available forecast horizons tau we have results for
inputs = ["FTLE", "Fourier"]          # available input feature types

dirs = {
    (inp, tau): f"outputs/tau_{tau}/{inp}/results"
    for inp in inputs
    for tau in taus
}

print("\n # ====================================== # ")
print("       [INFO] Loading prediction results     ")
print(" # ====================================== # ")
print(f"[INFO] Available inputs : {inputs}")
print(f"[INFO] Available taus   : {taus}")
print(f"[INFO] Total cases      : {len(dirs)} (input, tau) folders")



# ============================================================
# (2) Generate the time-series comparison figure
# ------------------------------------------------------------
# This will load pred.npy and true.npy for the selected cases
# and create a multi-panel plot comparing:
#   Ground Truth vs Prediction
# for the specified (inputs_to_plot, taus_to_plot).
# The figure is saved to save_dir/filename.
# ============================================================
print("\n # ====================================== # ")
print("   [INFO] Time-Series Comparison Plot       ")
print("         (Prediction vs Ground Truth)       ")
print(" # ====================================== # ")

tsf_comparison(
    dirs=dirs,
    taus=(10, 15),
    inputs=("FTLE", "Fourier"),
    save_dir="outputs/plots_for_manuscript/time_series",
    filename="Pred_vs_True.png",
    clip_negative=True
)




# ============================================================
# (3) Generate binary-metrics summary plots
# ------------------------------------------------------------
# After predictions are saved (pred.npy / true.npy) for each (input, tau),
# we compute and plot key metrics as a function of forecast horizon τ:
#   - F1 score (extreme-event classification after thresholding)
#   - PR–AUC (threshold-free ranking performance for extremes)
#   - alpha* (adjusted PR–AUC improvement above baseline event rate)
#   - |DeltaN_EE| (difference in counted extreme events between true and pred)
# The resulting 2×2 figure is saved to: save_dir/filename.
# ============================================================
print("\n # ====================================== # ")
print("   [INFO] Plot metrics vs tau              ")
print(" # ====================================== # ")

plot_metrics(
    dirs=dirs,
    taus=taus,
    inputs=inputs,
    thres=2.0,
    save_dir="outputs/plots_for_manuscript/Binary_metrics", 
    filename="F1_AUC_AlphaStar_DeltaN_vs_tau.png",
)