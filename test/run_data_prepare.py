"""Run data prepare and save to test_fixed.mat."""
import numpy as np
from scipy.io import savemat
from data_prepare_pv_es_ev import data_prepare_pv_es_ev

# Regenerate test.mat
print("Generating test_fixed.mat...")
data = data_prepare_pv_es_ev(day_price=21)

# Print key shapes for debugging
print("\nKey parameter shapes:")
print(f"  NOFDER: {data['NOFDER']}")
print(f"  NOFEV: {data['NOFEV']}")
print(f"  eta_ch shape: {data['param_std']['eta_ch'].shape}")
print(f"  eta_dis shape: {data['param_std']['eta_dis'].shape}")
print(f"  energy_init shape: {data['param_std']['energy_init'].shape}")

# Save to test_fixed.mat
mat_file = 'test_fixed.mat'
savemat(mat_file, data)
print(f"\nSaved to {mat_file}")
