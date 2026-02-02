import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc
from scipy.signal import find_peaks


def _to_1d_last_horizon(arr):
    """
    Convert predictions/targets to a 1D series.
    Expected shapes:
      - (N,) -> unchanged
      - (N, T) -> take last time step arr[:, -1]
      - (N, T, 1) -> take last time step arr[:, -1, 0]
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, -1]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, -1, 0]
    raise ValueError(f"Unsupported array shape: {arr.shape}")
    
    

def compute_f1(y_true, y_pred, thres, clip_pred_min=None):
    """
    Compute F1 score after binarizing extremes using *separate* thresholds:

        thr_true = mean(y_true) + thres * std(y_true)
        thr_pred = mean(y_pred) + thres * std(y_pred)

    Parameters
    ----------
    y_true : array-like
        Ground truth values (any shape; will be flattened).
    y_pred : array-like
        Predicted values (any shape; will be flattened).
    thres : float
        Threshold multiplier (number of sigmas above mean).
    clip_pred_min : float or None
        If not None, clip predictions below this value (e.g. 0.0).

    Returns
    -------
    f1 : float
        F1 score (np.nan if degenerate).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if clip_pred_min is not None:
        y_pred = np.clip(y_pred, a_min=clip_pred_min, a_max=None)

    # Remove NaN/inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return np.nan, (np.nan, np.nan)

    thr_true = np.mean(y_true) + thres * np.std(y_true)
    thr_pred = np.mean(y_pred) + thres * np.std(y_pred)

    y_true_bin = (y_true >= thr_true).astype(int)
    y_pred_bin = (y_pred >= thr_pred).astype(int)

    # Degenerate cases: all zeros or all ones
    if np.unique(y_true_bin).size < 2 or np.unique(y_pred_bin).size < 2:
        return np.nan, (thr_true, thr_pred)

    f1 = f1_score(y_true_bin, y_pred_bin)
    
    return f1
    
    

def compute_pr_auc(y_true, y_pred, thres, clip_pred_min=None):
    """
    Compute PR–AUC for extreme-event detection.
    
    1) Define "extreme" events from the ground truth:
       thr_true = mean(y_true) + thres * std(y_true)
       y_true_bin = 1 if y_true >= thr_true else 0
    
    2) Treat y_pred as a continuous score (higher => more likely extreme) and compute
       the precision–recall curve, then integrate it to get PR–AUC.
       (We do not threshold y_pred because PR–AUC evaluates performance across all
       possible score thresholds.)
    
    Returns
    -------
    pr_auc : float
        Area under the precision–recall curve (np.nan if y_true_bin has only one class).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if clip_pred_min is not None:
        y_pred = np.clip(y_pred, a_min=clip_pred_min, a_max=None)
        
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    thr_true = np.mean(y_true) + thres * np.std(y_true)
    y_true_bin = (y_true >= thr_true).astype(int)


    if len(np.unique(y_true_bin)) < 2:
        return np.nan, thr_true

    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred)
    pr_auc = auc(recall, precision)  
    
    return pr_auc



def compute_alpha_star(y_true, y_pred, n_quantiles, quantile_range):
    """
    Compute α* (Guth & Sapsis, 2019): the maximum PR–AUC improvement above baseline
    over a range of extreme-event thresholds.

    Procedure
    ---------
    For each quantile q in quantile_range:
      1) Define an extreme-event threshold on the TRUE signal:
             thr(q) = quantile(y_true, q)
         and create binary labels:
             y_true_bin = 1{y_true >= thr(q)}

      2) Let ω(q) be the event rate (fraction of extremes):
             ω(q) = mean(y_true_bin)

      3) Compute PR–AUC(q) using y_pred as continuous scores.

      4) Compute the "skill above chance":
             α(q) = PR–AUC(q) - ω(q)

    Output
    ------
    α* is the maximum α(q) over q, and ω* is the event rate at which it occurs.

    Returns
    -------
    alpha_star : float
        max_q [ PR–AUC(q) - ω(q) ]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    quantiles = np.linspace(quantile_range[0], quantile_range[1], n_quantiles)
    auc_values, omega_values = [], []

    for q in quantiles:
        thr = np.quantile(y_true, q)
        y_true_bin = (y_true >= thr).astype(int)
        omega = y_true_bin.mean()

        if omega <= 0.0 or omega >= 1.0:
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin, y_pred)
        auc_val = auc(recall, precision)

        auc_values.append(auc_val)
        omega_values.append(omega)

    if len(omega_values) == 0:
        return np.nan, np.nan  # alpha_star, omega_opt

    auc_values = np.asarray(auc_values)
    omega_values = np.asarray(omega_values)

    alpha_values = auc_values - omega_values
    idx = np.argmax(alpha_values)

    return alpha_values[idx], omega_values[idx]




def count_extreme_events(signal, dt, threshold_sigma, f_EE=None):
    """
    Count local maxima exceeding threshold = mean + threshold_sigma*std.
    Enforce minimum separation based on estimated or provided f_EE.
    """
    signal = np.asarray(signal)
    n = len(signal)
    time = np.arange(n) * dt

    mu, sigma = np.mean(signal), np.std(signal)
    threshold = mu + threshold_sigma * sigma

    peaks_initial, _ = find_peaks(signal, height=threshold)

    if len(peaks_initial) > 1 and f_EE is None:
        delta_t = np.diff(time[peaks_initial])
        T_EE = np.median(delta_t)
        f_EE = 1.0 / T_EE
    elif f_EE is None:
        f_EE = 1.0 / (0.05 * n * dt)
        T_EE = 1.0 / f_EE
    else:
        T_EE = 1.0 / f_EE

    min_separation = max(1, int(T_EE / dt))
    peaks_idx, _ = find_peaks(signal, height=threshold, distance=min_separation)

    return len(peaks_idx), f_EE



def compute_metrics(
    data,
    taus,
    inputs,
    thres,
    dt,
    alpha_n_quantiles,
    alpha_quantile_range,
    verbose=False,
):
    results = {
        "f1": {inp: [] for inp in inputs},
        "pr_auc": {inp: [] for inp in inputs},
        "alpha_star": {inp: [] for inp in inputs},
        "delta_N": {inp: [] for inp in inputs},
    }

    for inp in inputs:
        for tau in taus:
            if (inp, tau) not in data:
                for k in results:
                    results[k][inp].append(np.nan)
                if verbose:
                    print(f"[WARN] Missing data for ({inp}, tau={tau})")
                continue

            pred = _to_1d_last_horizon(data[(inp, tau)]["pred"])
            true = _to_1d_last_horizon(data[(inp, tau)]["true"])

            pred = np.clip(pred, a_min=0, a_max=None)

            f1_val = compute_f1(true, pred, thres=thres)
            pr_auc_val = compute_pr_auc(true, pred, thres=thres)
            alpha_star_val, _ = compute_alpha_star(
                true, pred,
                n_quantiles=alpha_n_quantiles,
                quantile_range=alpha_quantile_range
            )

            N_true, f_EE = count_extreme_events(true, dt=dt, threshold_sigma=thres, f_EE=None)
            N_pred, _    = count_extreme_events(pred, dt=dt, threshold_sigma=thres, f_EE=f_EE)
            delta_N = abs(N_true - N_pred)

            results["f1"][inp].append(f1_val)
            results["pr_auc"][inp].append(pr_auc_val)
            results["alpha_star"][inp].append(alpha_star_val)
            results["delta_N"][inp].append(delta_N)

            if verbose:
                print(f"[INFO] {inp} tau={tau:>2} | F1={f1_val:.3f} | PR-AUC={pr_auc_val:.3f} | a*={alpha_star_val:.3f} | dN={delta_N}")

    return results