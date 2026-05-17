#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Updater for Meta-Surface Dataset
--------------------------------------
功能：
1. 扫描指定目录的新 .mat 文件 (Result2)。
2. 智能追加到 HDF5：
   - 自动识别 ID 对应的 Source Z 位置 (580nm vs 880nm)。
   - 自动计算 Phase Term。
   - 解决 (N,2) vs (N,3) 的 cond 维度冲突问题。
"""

import os
import glob
import re
import h5py
import numpy as np
import torch
from tqdm import tqdm

# 引入基础工具 (请确保 data_builder.py 在同级目录)
from data_builder import (
    get_padded_size,
    torch_pad_center_circX_repZ,
    pad_coords_linear,
    compute_sdf_on_padded_geometry
)

# ================= 配置区域 =================
# 输入输出路径
SOURCE_DIR = "/Volumes/陈铭潜妙妙屋/FDTD_dataset/Result3/Ag Au Al SiO2/*.mat"
TARGET_H5 = "/Volumes/陈铭潜妙妙屋/FDTD_dataset/h5/unified_v6p3_260116.h5"

# 物理常数
Z0 = 376.73
PAD_MULTIPLE = 32

# Z轴位置规则
# 默认: 580nm
# 特殊: 2001-2020 -> 880nm
Z_DEFAULT = 580e-9
Z_SPECIAL = 880e-9
SPECIAL_ID_RANGE = [3031,3035,3041,3044,4006,4007,4008,
                    4009,4010,4017,4020,
                    5006,5007,5008,5009,5010,5021,5022,
                    5023,5024,5025,5026,5027,5028,5029,5030]
# ===========================================

def extract_id_from_filename(fname):
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else -1

def get_z_source(design_id):
    """根据 Design ID 判断光源 Z 位置"""
    if design_id in SPECIAL_ID_RANGE:
        return Z_SPECIAL
    return Z_DEFAULT

def process_and_append(fpath, h5_root):
    fname = os.path.basename(fpath)
    design_id = extract_id_from_filename(fname)

    # 1. 读取 MAT 文件
    try:
        f = h5py.File(fpath, 'r')
    except Exception as e:
        print(f"❌ 无法读取 {fname}: {e}")
        return

    if 'x' not in f or 'z' not in f:
        print(f"⚠️  跳过 {fname}: 数据不完整")
        f.close(); return

    # --- A. 基础几何处理 (复用 data_builder 逻辑) ---
    x_raw = f['x'][()].flatten()
    z_raw = f['z'][()].flatten()
    H, W = len(z_raw), len(x_raw)
    period_val = float(x_raw.max() - x_raw.min() + (x_raw[1]-x_raw[0]))

    tgt_h, tgt_w = get_padded_size(H, W, PAD_MULTIPLE)
    group_key = f"size_{tgt_h}_{tgt_w}"

    if 'RI_Gd' in f:
        ri_gd = f['RI_Gd'][()].astype(np.float32)
    else:
        ri_temp = f['RI']['real'][0,:,0,:]
        ri_gd = np.array(ri_temp, dtype=np.float32)

    ri_gd_padded = torch_pad_center_circX_repZ(ri_gd[None, ...], tgt_h, tgt_w)[0]
    sdf_padded = compute_sdf_on_padded_geometry(ri_gd_padded)

    x_pad = pad_coords_linear(x_raw, tgt_w)
    z_pad = pad_coords_linear(z_raw, tgt_h)
    xx, zz = np.meshgrid(x_pad, z_pad)

    mask_map = np.zeros((tgt_h, tgt_w), dtype='uint8')
    pad_t, pad_l = (tgt_h - H)//2, (tgt_w - W)//2
    mask_map[pad_t:pad_t+H, pad_l:pad_l+W] = 1

    lambdas = f['lambda'][()].flatten()
    N = len(lambdas)

    # --- B. 准备 HDF5 组结构 (关键修改：支持 3 列 cond) ---
    if group_key not in h5_root:
        grp = h5_root.create_group(group_key)
        grp.create_dataset('x', shape=(0, 5, tgt_h, tgt_w), maxshape=(None, 5, tgt_h, tgt_w), dtype='f4', compression="gzip")
        grp.create_dataset('y', shape=(0, 6, tgt_h, tgt_w), maxshape=(None, 6, tgt_h, tgt_w), dtype='f4', compression="gzip")
        grp.create_dataset('mask', shape=(0, 1, tgt_h, tgt_w), maxshape=(None, 1, tgt_h, tgt_w), dtype='u1', compression="gzip")
        # 直接创建 3 列的 cond: [k0_norm, k0p, phase]
        grp.create_dataset('cond', shape=(0, 3), maxshape=(None, 3), dtype='f4')
        grp.create_dataset('p', shape=(0, 1), maxshape=(None, 1), dtype='f4')
        grp.create_dataset('id', shape=(0, 1), maxshape=(None, 1), dtype='i4')

    grp = h5_root[group_key]

    # --- C. 维度冲突检查与升级 ---
    # 如果现有 cond 是 2 列，必须先升级为 3 列，否则无法追加新数据
    if grp['cond'].shape[1] == 2:
        print(f"🔧 [自动修复] 正在升级组 {group_key} 的 cond 维度 (2->3) ...")
        old_cond = grp['cond'][()]
        old_ids = grp['id'][()].flatten()

        # 回溯计算旧数据的 Phase
        z_vals = np.array([get_z_source(i) for i in old_ids], dtype=np.float32)
        k0_real = (old_cond[:, 0] + 1.0) / 1e-7
        phase_col = k0_real * z_vals

        new_cond_full = np.column_stack([old_cond, phase_col]).astype(np.float32)

        del grp['cond']
        grp.create_dataset('cond', data=new_cond_full, maxshape=(None, 3), dtype='f4')

    # --- D. 批量处理与追加 ---
    curr_len = grp['x'].shape[0]

    # 调整大小
    for key in ['x', 'y', 'mask', 'cond', 'p', 'id']:
        grp[key].resize(curr_len + N, axis=0)

    # 确定 Z Source
    z_source = get_z_source(design_id)

    proc_chunk = 50
    for i in range(0, N, proc_chunk):
        end = min(i+proc_chunk, N)
        size = end - i
        sl = slice(i, end)
        idx_slice = slice(curr_len+i, curr_len+end)

        # 1. Materials
        n_raw = f['RI']['real'][sl, :, 0, :]
        k_raw = f['RI']['imag'][sl, :, 0, :]
        mat_stack = np.stack([n_raw, k_raw], axis=1)
        mat_padded = torch_pad_center_circX_repZ(mat_stack, tgt_h, tgt_w)

        # 2. Fields
        ex_r, ex_i = f['Ex']['real'][sl,:,0,:], f['Ex']['imag'][sl,:,0,:]
        ez_r, ez_i = f['Ez']['real'][sl,:,0,:], f['Ez']['imag'][sl,:,0,:]
        hy_r, hy_i = f['Hy']['real'][sl,:,0,:], f['Hy']['imag'][sl,:,0,:]
        field_stack = np.stack([ex_r, ex_i, ez_r, ez_i, hy_r, hy_i], axis=1)
        field_padded = torch_pad_center_circX_repZ(field_stack, tgt_h, tgt_w)
        field_padded[:, 4:6] *= Z0 # Z0 Scaling

        # 3. Physics Inputs (X)
        n_p, k_p = mat_padded[:,0], mat_padded[:,1]
        eps_r = n_p**2 - k_p**2
        eps_i = 2 * n_p * k_p

        lams = lambdas[sl].reshape(-1, 1, 1)
        k0 = (2 * np.pi / lams).astype(np.float32)
        k0x = xx[None, ...] * k0
        k0z = zz[None, ...] * k0
        sdf_batch = np.repeat(sdf_padded[None, ...], size, axis=0)

        x_out = np.stack([eps_r, eps_i, k0x, k0z, sdf_batch], axis=1)

        # 4. Condition Vector (Cond) [3 columns]
        k0_val = k0.flatten()
        k0_norm = k0_val * 1e-7 - 1.0
        k0p_term = k0_val * period_val * 0.5
        phase_term = k0_val * z_source # New Phase Term

        c_out = np.stack([k0_norm, k0p_term, phase_term], axis=1)

        # 5. Others
        mask_batch = np.repeat(mask_map[None, None, ...], size, axis=0)
        p_out = np.full((size, 1), period_val, dtype=np.float32)
        id_out = np.full((size, 1), design_id, dtype=np.int32)

        # Write
        grp['x'][idx_slice] = x_out
        grp['y'][idx_slice] = field_padded
        grp['mask'][idx_slice] = mask_batch
        grp['cond'][idx_slice] = c_out
        grp['p'][idx_slice] = p_out
        grp['id'][idx_slice] = id_out

    f.close()
    print(f"   + Group [{group_key}]: 追加 {fname} (ID {design_id}, Zs={z_source*1e9:.0f}nm)")

def main():
    print(f"🚀 启动智能增量更新...")
    print(f"   源目录: {SOURCE_DIR}")
    print(f"   目标库: {TARGET_H5}")

    # 1. 获取已有 ID
    existing_ids = set()
    if os.path.exists(TARGET_H5):
        with h5py.File(TARGET_H5, 'r') as f:
            for k in f.keys():
                if 'id' in f[k]:
                    existing_ids.update(f[k]['id'][()].flatten().tolist())

    print(f"📊 现有设计 ID: {len(existing_ids)} 个")

    # 2. 扫描新文件
    files = sorted(glob.glob(SOURCE_DIR))
    new_files = []
    for f in files:
        fid = extract_id_from_filename(os.path.basename(f))
        if fid not in existing_ids:
            new_files.append(f)

    if not new_files:
        print("✅ 没有发现新文件。")
        return

    print(f"📦 准备处理 {len(new_files)} 个新文件...")

    # 3. 处理并追加
    with h5py.File(TARGET_H5, 'a') as h5_root:
        for fpath in tqdm(new_files):
            process_and_append(fpath, h5_root)

    print("🎉 全部更新完成！")

if __name__ == "__main__":
    main()