import numpy as np


# Compatibility shim for older dependencies (for example h5py 2.x)
# that still expect the removed numpy.typeDict alias.
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

# NumPy removed a set of old aliases that this codebase still uses.
if not hasattr(np, "bool"):
    np.bool = np.bool_
