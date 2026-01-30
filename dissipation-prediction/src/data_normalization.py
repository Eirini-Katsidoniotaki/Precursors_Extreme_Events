import numpy as np

class DataNormalizer:
    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self._fitted = False

    def fit(self, X, y):
        """Compute min and max for X and y."""
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        self.y_min = y.min()
        self.y_max = y.max()
        self._fitted = True

    def transform(self, X, y=None):
        """Normalize X and y to [0,1]."""
        if not self._fitted:
            raise RuntimeError("DataNormalizer must be fitted before transform().")
        
        Xn = (X - self.x_min) / (self.x_max - self.x_min + 1e-8)
        if y is not None:
            yn = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
            #print('X and y normalized to [0,1].')
            return Xn, yn
        #print('X normalized to [0,1].')
        return Xn

    def inverse_transform(self, Xn=None, yn=None):
        """Inverse transform from [0,1] back to original scale."""
        if not self._fitted:
            raise RuntimeError("DataNormalizer must be fitted before inverse_transform().")
        
        X, y = None, None
        if Xn is not None:
            X = Xn * (self.x_max - self.x_min) + self.x_min
        if yn is not None:
            y = yn * (self.y_max - self.y_min) + self.y_min
        return X, y

    def stats(self):
        """Return dictionary with min/max values."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max
        }
