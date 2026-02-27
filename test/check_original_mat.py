from scipy.io import loadmat
import numpy as np

mat = loadmat('data_prepare/param_day_21.mat')
print("=== Keys in original param_day_21.mat ===")
for key in sorted(mat.keys()):
    if not key.startswith('__'):
        val = mat[key]
        if isinstance(val, np.ndarray):
            print(f"{key}: shape {val.shape}, dtype {val.dtype}")
        else:
            print(f"{key}: {type(val)}")
