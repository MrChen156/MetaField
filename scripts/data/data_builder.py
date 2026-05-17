#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Unified Data Engineering (Final V6.3 - Production)
----------------------------------------------------------
Input:  .mat files in "source/Result1/*.mat"
Output: processed_data/unified_v6p3.h5 (grouped by Padded Size).

Features:
1. Topology: Center Aligned, X-Circular, Z-Replicate (via robust tensor expansion).
2. Physics: Explicit Period storage, No normalization burn-in.
3. Metadata: Stores 'design_id' for each sample (e.g. 11 from 0011.mat).
4. Storage: Compressed HDF5 groups.
"""

import os
import glob
import re
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# ================= CONFIG =================
PAD_MULTIPLE = 32
Z0 = 376.73
TARGET_DIR = "processed_data"
TARGET_H5 = os.path.join(TARGET_DIR, "unified_v6p3.h5")
# ==========================================

def get_padded_size(h, w, mult=32):
    th = ((h - 1) // mult + 1) * mult
    tw = ((w - 1) // mult + 1) * mult
    return th, tw

def torch_pad_center_circX_repZ(data, tgt_h, tgt_w):
    """
    Robust Padding: Center -> Z Replicate -> X Circular.
    Supports arbitrary leading dimensions (..., H, W).
    """
    if isinstance(data, np.ndarray):
        t = torch.from_numpy(data)
    else:
        t = data

    # Input shape handling
    input_shape = list(t.shape)
    h, w = input_shape[-2], input_shape[-1]

    # Calculate Padding
    diff_h, diff_w = tgt_h - h, tgt_w - w
    pad_t, pad_l = diff_h // 2, diff_w // 2
    pad_b, pad_r = diff_h - pad_t, diff_w - pad_l

    # Valid Core Indices
    top, bot = pad_t, pad_t + h
    left, right = pad_l, pad_l + w

    # 1. Create Output Canvas
    out_shape = input_shape[:-2] + [tgt_h, tgt_w]
    out = torch.zeros(out_shape, dtype=t.dtype)

    # Place Core
    out[..., top:bot, left:right] = t

    # 2. Z-Axis (Vertical) - Replicate
    if pad_t > 0:
        row_src = out[..., top:top+1, left:right] # (..., 1, W)
        # Construct expansion shape manually to be safe
        expand_dims = list(row_src.shape); expand_dims[-2] = pad_t
        out[..., :top, left:right] = row_src.expand(expand_dims)

    if pad_b > 0:
        row_src = out[..., bot-1:bot, left:right]
        expand_dims = list(row_src.shape); expand_dims[-2] = pad_b
        out[..., bot:, left:right] = row_src.expand(expand_dims)

    # At this point, the vertical strip [:, left:right] is fully filled.

    # 3. X-Axis (Horizontal) - Circular (wrap based on CORE strip)
    strip = out[..., :, left:right]  # (..., tgt_h, w)

    if pad_l > 0:
        out[..., :, :left] = strip[..., :, -pad_l:]      # left takes from core's right end
    if pad_r > 0:
        out[..., :, right:] = strip[..., :, :pad_r]      # right takes from core's left end

    return out.numpy()

def pad_coords_linear(arr, tgt_len):
    """ Linear extension for 1D coordinate array (Center Aligned) """
    l = len(arr)
    diff = tgt_len - l
    pad_l = diff // 2
    pad_r = diff - pad_l

    dx = (arr[1] - arr[0]) if len(arr) > 1 else 1.0

    prefix = arr[0] - dx * np.arange(pad_l, 0, -1)
    suffix = arr[-1] + dx * np.arange(1, pad_r + 1)

    return np.concatenate([prefix, arr, suffix]).astype(np.float32)

def compute_sdf_on_padded_geometry(ri_gd_padded):
    dy, dx = np.gradient(ri_gd_padded)
    grad_mag = np.sqrt(dx**2 + dy**2)
    is_edge = (grad_mag > 1e-3)

    if is_edge.sum() == 0:
        return np.ones_like(ri_gd_padded, dtype=np.float32)

    dist = distance_transform_edt(np.logical_not(is_edge))
    # Normalize by roughly 1/4 width (Period/4)
    return (dist / (ri_gd_padded.shape[1] / 4.0)).astype(np.float32)

def extract_design_id(filename):
    # e.g. "0011.mat" -> 11
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1 # Unknown

def process_file(fpath, f, h5_root):
    fname = os.path.basename(fpath)
    design_id = extract_design_id(fname)

    # 1. Read Raw Info
    if 'x' not in f or 'z' not in f: return
    x_raw = f['x'][()].flatten()
    z_raw = f['z'][()].flatten()
    H, W = len(z_raw), len(x_raw)

    # Period (Physical) - calculated from RAW data
    period_val = float(x_raw.max() - x_raw.min() + (x_raw[1]-x_raw[0]))

    # 2. Determine Target Size (Group Key)
    tgt_h, tgt_w = get_padded_size(H, W, PAD_MULTIPLE)
    group_key = f"size_{tgt_h}_{tgt_w}"

    # 3. Geometry & SDF
    if 'RI_Gd' in f:
        ri_gd = f['RI_Gd'][()].astype(np.float32)
    else:
        ri_temp = f['RI']['real'][0,:,0,:]
        ri_gd = np.array(ri_temp, dtype=np.float32)

    # Pad Geometry & Compute SDF
    ri_gd_padded = torch_pad_center_circX_repZ(ri_gd[None, ...], tgt_h, tgt_w)[0]
    sdf_padded = compute_sdf_on_padded_geometry(ri_gd_padded)

    # 4. Coords (Linear Ext)
    x_pad = pad_coords_linear(x_raw, tgt_w)
    z_pad = pad_coords_linear(z_raw, tgt_h)
    xx, zz = np.meshgrid(x_pad, z_pad) # (TgtH, TgtW)

    # 5. Create Mask
    mask_map = np.zeros((tgt_h, tgt_w), dtype='uint8')
    pad_t, pad_l = (tgt_h - H)//2, (tgt_w - W)//2
    mask_map[pad_t:pad_t+H, pad_l:pad_l+W] = 1

    # 6. Dynamic Data Loop
    lambdas = f['lambda'][()].flatten()
    N = len(lambdas)

    # H5 Dataset Creation
    if group_key not in h5_root:
        grp = h5_root.create_group(group_key)
        # x: [eps_r, eps_i, k0x, k0z, idf]
        grp.create_dataset('x', shape=(0, 5, tgt_h, tgt_w), maxshape=(None, 5, tgt_h, tgt_w), dtype='f4', compression="gzip")
        # y: Fields
        grp.create_dataset('y', shape=(0, 6, tgt_h, tgt_w), maxshape=(None, 6, tgt_h, tgt_w), dtype='f4', compression="gzip")
        # mask: Binary
        grp.create_dataset('mask', shape=(0, 1, tgt_h, tgt_w), maxshape=(None, 1, tgt_h, tgt_w), dtype='u1', compression="gzip")
        # cond: [k0_norm, k0p_term]
        grp.create_dataset('cond', shape=(0, 2), maxshape=(None, 2), dtype='f4')
        # p: Period
        grp.create_dataset('p', shape=(0, 1), maxshape=(None, 1), dtype='f4')
        # id: Design ID
        grp.create_dataset('id', shape=(0, 1), maxshape=(None, 1), dtype='i4')

    grp = h5_root[group_key]
    curr_len = grp['x'].shape[0]

    # Resize
    for key in ['x', 'y', 'mask', 'cond', 'p', 'id']:
        grp[key].resize(curr_len + N, axis=0)

    # Batch Process
    proc_chunk = 50
    for i in range(0, N, proc_chunk):
        end = min(i+proc_chunk, N)
        size = end - i
        sl = slice(i, end)

        # A. Materials
        n_raw = f['RI']['real'][sl, :, 0, :]
        k_raw = f['RI']['imag'][sl, :, 0, :]
        mat_stack = np.stack([n_raw, k_raw], axis=1) # (B, 2, H, W)

        # Pad Batch
        mat_padded = torch_pad_center_circX_repZ(mat_stack, tgt_h, tgt_w)

        # B. Fields
        ex_r, ex_i = f['Ex']['real'][sl,:,0,:], f['Ex']['imag'][sl,:,0,:]
        ez_r, ez_i = f['Ez']['real'][sl,:,0,:], f['Ez']['imag'][sl,:,0,:]
        hy_r, hy_i = f['Hy']['real'][sl,:,0,:], f['Hy']['imag'][sl,:,0,:]
        field_stack = np.stack([ex_r, ex_i, ez_r, ez_i, hy_r, hy_i], axis=1)

        field_padded = torch_pad_center_circX_repZ(field_stack, tgt_h, tgt_w)

        # C. Assemble Physics
        # Eps
        n_p, k_p = mat_padded[:,0], mat_padded[:,1]
        eps_r = n_p**2 - k_p**2
        eps_i = 2 * n_p * k_p

        # K0 Coords
        lams = lambdas[sl].reshape(-1, 1, 1)
        k0 = (2 * np.pi / lams).astype(np.float32)
        k0x = xx[None, ...] * k0
        k0z = zz[None, ...] * k0

        # SDF Broadcast
        sdf_batch = np.repeat(sdf_padded[None, ...], size, axis=0)

        # X: [eps_r, eps_i, k0x, k0z, sdf]
        x_out = np.stack([eps_r, eps_i, k0x, k0z, sdf_batch], axis=1)

        # Y: Z0 scale
        field_padded[:, 4:6] *= Z0

        # Mask
        mask_batch = np.repeat(mask_map[None, None, ...], size, axis=0)

        # Cond
        k0_val = k0.flatten()
        k0_norm = k0_val * 1e-7 - 1.0
        k0p_term = k0_val * period_val * 0.5 # Using raw period
        c_out = np.stack([k0_norm, k0p_term], axis=1)

        # Period & ID
        p_out = np.full((size, 1), period_val, dtype=np.float32)
        id_out = np.full((size, 1), design_id, dtype=np.int32)

        # Write
        idx_slice = slice(curr_len+i, curr_len+end)
        grp['x'][idx_slice] = x_out
        grp['y'][idx_slice] = field_padded
        grp['mask'][idx_slice] = mask_batch
        grp['cond'][idx_slice] = c_out
        grp['p'][idx_slice] = p_out
        grp['id'][idx_slice] = id_out

    print(f"   + Group [{group_key}]: Added {fname} (ID {design_id})")

def main():
    parser = argparse.ArgumentParser()
    # Default to your folder structure
    parser.add_argument('--src', type=str, default='source/Result1/*.mat')
    args = parser.parse_args()

    files = sorted(glob.glob(args.src))
    print(f"🚀 Processing {len(files)} files into {TARGET_H5} ...")

    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.exists(TARGET_H5):
        os.remove(TARGET_H5) # Fresh start

    with h5py.File(TARGET_H5, 'w') as h5_out:
        for fpath in tqdm(files):
            try:
                with h5py.File(fpath, 'r') as f_in:
                    process_file(fpath, f_in, h5_out)
            except Exception as e:
                print(f"❌ Error reading {fpath}: {e}")
                import traceback
                traceback.print_exc()

    print_summary()

def print_summary():
    print("\n📊 Database Summary:")
    if not os.path.exists(TARGET_H5): return
    with h5py.File(TARGET_H5, 'r') as f:
        total = 0
        for k in sorted(f.keys()):
            if 'x' in f[k]:
                n = f[k]['x'].shape[0]
                shape = f[k]['x'].shape[2:]
                print(f"   Group [{k:<16}]: {n:<6} samples | Shape {shape}")
                total += n
        print(f"✅ Total Samples: {total}")

if __name__ == '__main__':
    main()