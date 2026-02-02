import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_normalization import DataNormalizer

def load_data(args):
    """
    Load raw input–output data for the selected representation.

    This function handles reading the raw feature matrix X and target y,
    based on the input type specified in args.inp.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Expected to contain:
        - args.inp : str
            Input representation identifier ("FTLE" or "Fourier")
        - args.directory_data : str
            Path to directory containing the .npy data files

    Returns
    -------
    X : np.ndarray
        Raw input feature matrix
    y : np.ndarray
        Raw target array (energy dissipation)
    label : str
        String identifier of the input type (e.g. "FTLE", "Fourier")
    """
    
    data_dir = args.directory_data

    if args.inp == "FTLE":
        X_path = os.path.join(data_dir, "X_FTLE.npy")
        y_path = os.path.join(data_dir, "y_dissipation.npy")

        feature_description = [
            "X[:, 0] --> Leading FTLE",
            "X[:, 1] --> Time derivative of Leading FTLE",
        ]

        input_name = "FTLE"

    elif args.inp == "Fourier":
        X_path = os.path.join(data_dir, "X_Fourier.npy")
        y_path = os.path.join(data_dir, "y_dissipation.npy")

        feature_description = [
            "X[:, 0] --> Real part of a(1,0)",
            "X[:, 1] --> Imaginary part of a(1,0)",
        ]

        input_name = "Fourier mode a(1,0)"

    else:
        raise ValueError(
            f"Unknown input type '{args.inp}'. "
            "Expected 'FTLE' or 'Fourier'."
        )

    # ---- File checks
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")

    # ---- Load
    X = np.load(X_path)
    y = np.load(y_path)

    # ---- Info print (kept explicit and readable)
    print("\n # =============== Dataset Loaded ============== # ")
    print(f" Input type        : {input_name}")
    print(f" X shape           : {X.shape}")
    print(f" y shape           : {y.shape}  (Energy Dissipation)")
    print(" Feature mapping   : ")
    for line in feature_description:
        print(f"  {line}")
    print(" # =============================================== # ")

    return X, y, args.inp

    
   
    
    
def split_train_val_test(X, y, args, verbose=True):
    """
    Chronological split of time-series data into train/val/test
    using sizes defined in args.

    Expects args to have:
      - args.test_size
      - args.val_size
    """

    test_size = args.test_size
    val_size  = args.val_size

    assert 0 < test_size < 1
    assert 0 < val_size < 1
    assert test_size + val_size < 1
    assert len(X) == len(y), "X and y must have same length"

    N = len(X)
    n_test = int(N * test_size)
    n_val  = int(N * val_size)
    n_train = N - n_test - n_val

    # Chronological split
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val     = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test   = X[n_train + n_val:], y[n_train + n_val:]

    if verbose:
        #print("\n # ============================================ # ")
        print("            [INFO] Data split (chronological)")
        print(" # ============================================ # ")
        print(f" Train size : {X_train.shape}")
        print(f" Val size   : {X_val.shape}")
        print(f" Test size  : {X_test.shape}")
        

    return X_train, y_train, X_val, y_val, X_test, y_test
    
    



def create_sequences(X, y, m, h, label_len, to_torch=True):
        """
        Construct input–output sequences for supervised time-series learning.
    
        From a multivariate input signal X(t) and a scalar target y(t),
        this function generates sliding-window training samples of the form:
    
            input  : X[t-m : t]
            target : y[t-label_len : t+h]
    
        The target sequence includes a short history of the target signal
        (label window) followed by the future prediction horizon. This allows
        the model to condition its forecast on recent ground-truth target
        values rather than predicting from a cold start.
    
        Parameters
        ----------
        X : ndarray, shape (N, d)
            Input feature time series, where N is the number of time steps
            and d is the number of input features (e.g. FTLE components).
        y : ndarray, shape (N,)
            Target time series (e.g. energy dissipation).
        m : int
            Length of the input lookback window (number of past time steps).
        h : int
            Length of the prediction horizon (number of future time steps).
        label_len : int
            Length of the label window preceding the prediction horizon.
            Provides recent target history to the model during training.
        to_torch : bool, default True
            If True, return PyTorch tensors; otherwise return NumPy arrays.
    
        Returns
        -------
        X_seqs : ndarray or torch.Tensor, shape (N_samples, m, d)
            Input sequences consisting of m past time steps.
        y_seqs : ndarray or torch.Tensor, shape (N_samples, label_len + h, 1)
            Target sequences containing the label window and prediction horizon.
        """
    
        N = len(X)
        X_seqs, y_seqs = [], []
       

        for t in range(m, N - h):
            # past context
            X_window = X[t - m : t]     
            
            # Target sequence used during training:
            #   y[t-label_len : t] -> known target history (conditioning)
            #   y[t : t+h]         -> future values to be predicted
            #
            # At inference time, only the history up to t is provided;
            # the model predicts y[t : t+h] without access to future targets.
            
            y_window = y[t - label_len : t + h]     
            
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





def prepare_ml_data(X, y, m, h, label_len, args):
    """
    Prepare raw data for the ML pipeline.

    This function performs:
      1) Train / validation / test split
      2) Normalization (fit on train only)
      3) Sequence creation
      4) Batching into PyTorch DataLoaders

    Parameters
    ----------
    X : np.ndarray
        Raw input features
    y : np.ndarray
        Raw target values
    args : argparse.Namespace
        Experiment configuration


    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch DataLoaders for training, validation, and testing
    normalizer : DataNormalizer
        Fitted normalizer (useful for inverse transforms)
    """

    # ==================================================
    # Split data
    # ==================================================
    print("\n # ============================================ # ")
    print(" [INFO] Split data into train - validation - test:")
    #print(" # ============================================ # ")

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, args)

    # ==================================================
    # Normalize data (fit on train only)
    # ==================================================
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

    for name, (X_split, y_split) in datasets.items():
        print(f"[INFO] Normalizing {name} data...")
        normalized[name] = normalizer.transform(X_split, y_split)

    X_train_norm, y_train_norm = normalized["train"]
    X_val_norm,   y_val_norm   = normalized["val"]
    X_test_norm,  y_test_norm  = normalized["test"]
    
    datasets_norm = {
        "train": (X_train_norm, y_train_norm),
        "val":   (X_val_norm,   y_val_norm),
        "test":  (X_test_norm,  y_test_norm),
    }

    print("[INFO] Normalization statistics:")
    print(normalizer.stats())

    
    print("\n # ================================== # ")
    print("    [INFO] Create sequential datasets.   ")
    print(" # ================================== # ")


    print(f"\n [INFO] Lookback window = {args.lookback} s → {m} steps")
    print(f"\n [INFO] Prediction horizon tau = {args.tau} s → {h} steps")
    print(f"\n [INFO] Label length = {args.label} s → {label_len} steps")


    X_train_seq, y_train_seq = create_sequences(
        X_train_norm, y_train_norm, m, h, label_len, to_torch=True
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val_norm, y_val_norm, m, h, label_len, to_torch=True
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_norm, y_test_norm, m, h, label_len, to_torch=True
    )

    print("--- Dataset shapes after sequence creation ---")
    print(f"Train: X {X_train_seq.shape}, y {y_train_seq.shape}")
    print(f"Val  : X {X_val_seq.shape},   y {y_val_seq.shape}")
    print(f"Test : X {X_test_seq.shape},  y {y_test_seq.shape}")
    print("======================================================================= ")

    
    
    print("\n # ====================================== # ")
    print("   [INFO] Create batched datasets for ML.   ")
    print(" # ====================================== # ")

    train_loader = DataLoader(
        TensorDataset(X_train_seq, y_train_seq),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        TensorDataset(X_val_seq, y_val_seq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        TensorDataset(X_test_seq, y_test_seq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print("--- Dataset shapes after batch creation ---")
    print(f"[INFO] batch_size   : {args.batch_size}")
    print(f"[INFO] train_loader batches: {len(train_loader)}")
    print(f"[INFO] val_loader   batches: {len(val_loader)}")
    print(f"[INFO] test_loader  batches: {len(test_loader)}")
    print("======================================================================= ")

    return datasets, datasets_norm, train_loader, val_loader, test_loader, normalizer