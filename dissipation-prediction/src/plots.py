import os
import numpy as np
import matplotlib.pyplot as plt
from args import get_args

from src.metrics import compute_metrics



def load_pred_true(dirs):
    """
    dirs: dict like {('FTLE', 2): 'outputs/tau_2/FTLE/results', ...}
    returns: dict {(input_type, tau): {'pred': np.ndarray, 'true': np.ndarray}}
    """
    data = {}
    for (input_type, tau), path in dirs.items():
        pred_path = os.path.join(path, "pred.npy")
        true_path = os.path.join(path, "true.npy")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing pred file: {pred_path}")
        if not os.path.exists(true_path):
            raise FileNotFoundError(f"Missing true file: {true_path}")

        data[(input_type, tau)] = {
            "pred": np.load(pred_path),
            "true": np.load(true_path),
        }
    return data



def tsf_comparison(
    dirs,
    taus,
    inputs,
    save_dir,
    filename,
    clip_negative=True
):
    """
    Makes the 2x2 time-series plot comparing true vs pred for (inputs x taus).
    """
    data = load_pred_true(dirs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, tau in enumerate(taus):
        for j, input_type in enumerate(inputs):
            ax = axes[i, j]
            y_true = data[(input_type, tau)]["true"]
            y_pred = data[(input_type, tau)]["pred"]

            # Use last forecast horizon
            y_true_plot = y_true[:, -1, 0]
            y_pred_plot = y_pred[:, -1, 0]

            if clip_negative:
                y_pred_plot = np.clip(y_pred_plot, a_min=0, a_max=None)

            ax.plot(y_true_plot, label="Ground Truth", color = "blue", lw=1.2)
            ax.plot(y_pred_plot, label="Prediction",   color = "red",  lw=1.2, alpha=0.8)

            ax.set_title(f"{input_type} | τ = {tau}", fontsize=12, weight="semibold")

            if i == 1:
                ax.set_xlabel("Time", fontsize=10)
            if j == 0:
                ax.set_ylabel("Energy Dissipation", fontsize=10)

            ax.grid(alpha=0.3)

            # Legend only on upper-right subplot
            if i == 0 and j == 1:
                ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


    print(f"[INFO] Figure saved to: {save_path}")
    return save_path
    
    


def plot_metrics(dirs, taus, inputs, thres, save_dir, filename):
    
    args = get_args()
    
    # --- load pred/true first (from dirs) ---
    data = load_pred_true(dirs)


    # --- compute metrics ---
    results = compute_metrics(
        data=data,
        taus=taus,
        inputs=inputs,
        thres=thres,
        dt=args.DT,
        alpha_n_quantiles=50,
        alpha_quantile_range=(0.5, 0.99),
        verbose=False,
    )

    f1_results = results["f1"]
    pr_auc_results = results["pr_auc"]
    alpha_star_results = results["alpha_star"]
    delta_N_results = results["delta_N"]

    # plotting settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelpad": 4,
        "legend.frameon": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    markers = {'FTLE': 'o', 'Fourier': 's'}
    styles  = {'FTLE': '-', 'Fourier': '--'}
    colors  = {'FTLE': 'k', 'Fourier': 'dimgray'}

    axes = axes.flatten()
    letters = ['(a)', '(b)', '(c)', '(d)']

    # (a) F1
    ax = axes[0]
    for inp in inputs:
        ax.plot(taus, f1_results[inp], styles[inp], color=colors[inp],
                marker=markers[inp], markersize=4, linewidth=1.3, label=inp)
    ax.set_ylabel('F1 score')
    ax.set_ylim(0, 1.02)
    ax.text(0.85, 0.94, letters[0], transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left', handlelength=1.5)

    # (b) PR–AUC
    ax = axes[1]
    for inp in inputs:
        ax.plot(taus, pr_auc_results[inp], styles[inp], color=colors[inp],
                marker=markers[inp], markersize=4, linewidth=1.3)
    ax.set_ylabel('PR–AUC')
    ax.set_ylim(0, 1.02)
    ax.text(0.85, 0.94, letters[1], transform=ax.transAxes, fontsize=11, fontweight='bold')

    # (c) alpha*
    ax = axes[2]
    for inp in inputs:
        ax.plot(taus, alpha_star_results[inp], styles[inp], color=colors[inp],
                marker=markers[inp], markersize=4, linewidth=1.3)
    ax.set_xlabel(r'Forecast horizon $\tau$ [s]')
    ax.set_ylabel(r'$\alpha^*$')
    ax.set_ylim(0, 1.02)
    ax.text(0.85, 0.94, letters[2], transform=ax.transAxes, fontsize=11, fontweight='bold')

    # (d) |ΔN_EE|
    ax = axes[3]
    for inp in inputs:
        ax.plot(taus, delta_N_results[inp], styles[inp], color=colors[inp],
                marker=markers[inp], markersize=4, linewidth=1.3)
    ax.set_xlabel(r'Forecast horizon $\tau$ [s]')
    ax.set_ylabel(r'$|\Delta N_{\mathrm{EE}}|$')
    ax.text(0.85, 0.94, letters[3], transform=ax.transAxes, fontsize=11, fontweight='bold')

    plt.tight_layout()

    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')

    
    print(f"[INFO] 2×2 metrics figure saved to: {save_path}")
    return save_path
