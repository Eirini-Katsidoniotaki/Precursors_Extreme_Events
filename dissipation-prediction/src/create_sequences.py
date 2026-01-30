import numpy as np
import torch
from torch.utils.data import TensorDataset


class SequenceCreator:
    def __init__(self, m, h):
        """
        Parameters
        ----------
        m : int
            Number of past steps to include in each input sequence.
        h : int
            Forecast horizon (number of future steps to predict).
        """
        self.m  = m
        self.h  = h
        
        
    def create_sequences(self, X, y, label_len, to_torch=True):
        """
        Create input-output sequences for supervised learning.

        Parameters
        ----------
        X : ndarray, shape (N, d)
            Input features (e.g. FTLEs).
        y : ndarray, shape (N,)
            Target signal (e.g. dissipation).
        to_torch : bool, default False
            If True, return PyTorch tensors instead of NumPy arrays.

        Returns
        -------
        X_seq : Tensor, shape (N_samples, m, d)
            Input sequences of length m.
        y_seq : Tensor, shape (N_samples, h, 1)
            Target output sequences of length h.
        """
        N = len(X)
        X_seqs, y_seqs = [], []
       

        for t in range(self.m, N - self.h):
            # past context
            X_window = X[t - self.m : t]     # (m, d): past context
            
            # future horizon
            y_window = y[t - label_len : t + self.h]     # (label_len + h,): past + future horizon
            
            # --- Collect ---
            X_seqs.append(X_window)
            y_seqs.append(y_window[:, None]) # (h, 1)


        # Stack into arrays
        X_seqs, y_seqs = np.array(X_seqs), np.array(y_seqs)
        

        # Convert to torch
        if to_torch:
            #X_seqs = torch.from_numpy(X_seqs).float()
            #y_seqs = torch.from_numpy(y_seqs).float()
            
            X_seqs = torch.as_tensor(X_seqs, dtype=torch.float32)
            y_seqs = torch.as_tensor(y_seqs, dtype=torch.float32)
            
        return X_seqs, y_seqs
        


