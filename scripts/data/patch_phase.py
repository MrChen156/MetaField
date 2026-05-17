#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch: Inject Source Phase Information
--------------------------------------
Logic:
1. Parse existing 'cond' to recover real k0.
2. Identify Z_source based on Design ID.
   - IDs [1006, 1007, 1008, 1009, 1010] -> 880 nm
   - Others -> 580 nm
3. Calculate Phase Term = k0 * z_source.
4. Overwrite 'cond' dataset with new shape (N, 3).
"""

import h5py
import numpy as np
from tqdm import tqdm

H5_PATH = "processed_data/unified_v6p3.h5"

# Config
NEW_BATCH_IDS = {1006, 1007, 1008, 1009, 1010}
Z_OLD = 580e-9  # 580 nm -> meters
Z_NEW = 880e-9  # 880 nm -> meters

def patch_phase():
    print(f"🔧 Patching Phase Info into {H5_PATH} ...")

    with h5py.File(H5_PATH, 'r+') as f:
        # Iterate over all size groups
        for grp_key in tqdm(f.keys()):
            grp = f[grp_key]

            # 1. Check if already patched
            current_cond = grp['cond'][()]
            N, dim = current_cond.shape

            if dim == 3:
                print(f"⚠️  Group {grp_key} already has 3 dimensions. Skipping...")
                continue

            # 2. Get necessary data
            ids = grp['id'][()].flatten()  # Shape (N,)
            k0_norm = current_cond[:, 0]

            # Recover real k0 (Recall: k0_norm = k0 * 1e-7 - 1.0)
            k0_real = (k0_norm + 1.0) / 1e-7

            # 3. Determine Z for each sample
            z_vals = np.zeros_like(k0_real)

            # Vectorized assignment is hard with set lookup, using list comp
            # (Fast enough for dataset sizes < 1M)
            z_list = [Z_NEW if i in NEW_BATCH_IDS else Z_OLD for i in ids]
            z_arr = np.array(z_list, dtype=np.float32)

            # 4. Calculate Phase Term (rad)
            # phase = k0 * z
            phase_term = k0_real * z_arr

            # 5. Stack new condition vector
            # New shape: (N, 3) -> [k0_norm, k0p_term, phase_term]
            new_cond = np.column_stack([current_cond, phase_term]).astype(np.float32)

            # 6. Overwrite Dataset
            # Since we are changing dim 2 from 2 to 3, we must delete and recreate
            # UNLESS maxshape was set properly. To be safe, we delete/create.
            del grp['cond']

            # Create with maxshape=(None, None) to allow future row/col expansion
            dset = grp.create_dataset('cond', data=new_cond, maxshape=(None, None), dtype='f4', compression="gzip")

            # Verification
            # print(f"   Group {grp_key}: Updated cond to {dset.shape}")

    print("✅ Phase Patch Complete. All cond vectors are now (N, 3).")

if __name__ == "__main__":
    patch_phase()