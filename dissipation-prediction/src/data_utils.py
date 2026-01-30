import os
import numpy as np



def load_ftle(args):
    data_dir = args.directory_data

    X_path = os.path.join(data_dir, "X_FTLE.npy")
    y_path = os.path.join(data_dir, "y_dissipation.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)
    
    
    print("\n # === Dataset Loaded === # ")
    print(f" Input type        : FTLE ")
    print(f" X shape           : {X.shape} ")
    print(f" y shape           : {y.shape}  (Energy Dissipation) ")
    print(" Feature mapping   : ")
    print("  X[:, 0]        --> Leading FTLE ")
    print("  X[:, 1]        --> Time derivative of Leading FTLE ")
    print(" # ====================== # ")


    return X, y, "FTLE"
    
    

def load_fourier(args):
    data_dir = args.directory_data

    X_path = os.path.join(data_dir, "X_Fourier.npy")
    y_path = os.path.join(data_dir, "y_dissipation.npy")
    
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing {X_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)
    
    print("\n # === Dataset Loaded === # ")
    print(" Input type        : Fourier mode a(1,0) ")
    print(f" X shape           : {X.shape} ")
    print(f" y shape           : {y.shape}  (Energy Dissipation) ")
    print(" Feature mapping   : ")
    print("  X[:, 0]        --> Real part of a(1,0) ")
    print("  X[:, 1]        --> Imaginary part of a(1,0) ")
    print(" # ====================== # ")

    return X, y, "Fourier"
    
    
    
    
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
        print("\n # ============================================ # ")
        print("            [INFO] Data split (chronological)")
        print(" # ============================================ # ")
        print(f" Train size : {X_train.shape}")
        print(f" Val size   : {X_val.shape}")
        print(f" Test size  : {X_test.shape}")
        

    return X_train, y_train, X_val, y_val, X_test, y_test

