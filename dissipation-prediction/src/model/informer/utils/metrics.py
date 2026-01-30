import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe
    
    
def MSE_at_specific_step(preds, trues, t):
    """
    Compute MSE at a specific horizon t (0-based index).
    preds, trues: arrays of shape (N, H, 1)
    t: horizon index (0 = t+1, 99 = t+100)
    """
    pred_t = preds[:, t, 0]   # shape (N,)
    true_t = trues[:, t, 0]   # shape (N,)
    return np.mean((pred_t - true_t)**2)
    

def MAE_at_specific_step(preds, trues, t):
    pred_t = preds[:, t, 0]   # shape (N,)
    true_t = trues[:, t, 0]   # shape (N,)
    return np.mean(np.abs(pred_t-true_t))
