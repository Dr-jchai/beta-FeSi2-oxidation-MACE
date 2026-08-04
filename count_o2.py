#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终优化版：统计两个 XDATCAR 中的 remaining O₂ 百分比
- 一次性读取轨迹
- 采样间隔 10 步
- 只对 O 原子子集使用 cKDTree 计算邻居（高效）
- multiprocessing 并行处理两个轨迹
- 输出：o2_dual_evolution.csv（time_ps, Si-terminal_remaining_O2, Fe-terminal_remaining_O2）
"""

import numpy as np
import pandas as pd
from ase.io import read
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# ====================== 配置 ======================
FILES = {
    'Fe_terminal': "XDATCAR_Fe-terminal",
    'Si_terminal': "XDATCAR_Si-terminal"
}

DT_FS       = 1.0
O2_CUTOFF   = 1.5
INITIAL_O2  = 104
SAMPLE_STEP = 10                     # 采样间隔固定为 10
N_PROCESSES = max(1, cpu_count() - 1)   # 使用所有核心 -1

# ====================== 函数 ======================
def count_free_o2(pos_o):
    """仅对 O 原子使用 cKDTree 计算自由 O₂ 数量（最快方式）"""
    if len(pos_o) < 2:
        return 0
    tree = cKDTree(pos_o)
    pairs = tree.query_pairs(O2_CUTOFF, output_type='ndarray')
    
    degrees = np.zeros(len(pos_o), dtype=int)
    np.add.at(degrees, pairs[:, 0], 1)
    np.add.at(degrees, pairs[:, 1], 1)
    
    return np.sum(degrees == 1) // 2

def process_trajectory(args):
    """处理单个轨迹（并行函数）"""
    name, filepath = args
    print(f"[{name}] 开始读取轨迹...")
    start_t = time.time()
    trajectory = read(filepath, index=':')
    print(f"[{name}] 读取完成，用时 {time.time() - start_t:.2f} 秒 ({len(trajectory)} 帧)")

    time_ps_list = []
    perc_list = []
    print(f"[{name}] 开始统计 O₂（采样间隔 {SAMPLE_STEP}）...")

    for i in tqdm(range(0, len(trajectory), SAMPLE_STEP), desc=name):
        atoms = trajectory[i]
        # 只提取 O 原子的坐标（极大减少计算量）
        o_mask = np.array([atom.symbol == 'O' for atom in atoms])
        pos_o = atoms.positions[o_mask]
        
        num_o2 = count_free_o2(pos_o)
        perc = num_o2 / INITIAL_O2 * 100.0 if INITIAL_O2 > 0 else 0.0
        time_ps = i * DT_FS / 1000.0
        
        time_ps_list.append(time_ps)
        perc_list.append(perc)

    return name, time_ps_list, perc_list

# ====================== 主程序 ======================
if __name__ == '__main__':
    print(f"启动多进程处理（使用 {N_PROCESSES} 个进程）...")

    start_time = time.time()

    # 准备任务
    tasks = list(FILES.items())

    # 并行执行
    with Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(process_trajectory, tasks),
            total=len(tasks),
            desc="并行处理轨迹"
        ))

    # 合并结果
    data = {'time_ps': None, 
            'Si_terminal_remaining_O2': None, 
            'Fe_terminal_remaining_O2': None}

    for name, time_ps_list, perc_list in results:
        if data['time_ps'] is None:
            data['time_ps'] = time_ps_list
        if name == 'Si_terminal':
            data['Si_terminal_remaining_O2'] = perc_list
        elif name == 'Fe_terminal':
            data['Fe_terminal_remaining_O2'] = perc_list

    # 生成 DataFrame 并保存
    df = pd.DataFrame(data)
    df = df[['time_ps', 'Si_terminal_remaining_O2', 'Fe_terminal_remaining_O2']]
    df.columns = ['time_ps', 'Si-terminal_remaining_O2', 'Fe-terminal_remaining_O2']

    df.to_csv('o2_dual_evolution.csv', index=False, float_format='%.4f')
    print(f"\n已保存：o2_dual_evolution.csv （行数：{len(df)}）")

    # 后期统计（约最后 7000 原始步）
    late_df = df.tail((7000 // SAMPLE_STEP) + 1)
    print("\n后期平均剩余 O₂ 百分比（约最后 7000 原始步）：")
    print(f"Si-terminal: {late_df['Si-terminal_remaining_O2'].mean():.2f} % "
          f"(std = {late_df['Si-terminal_remaining_O2'].std():.2f})")
    print(f"Fe-terminal: {late_df['Fe-terminal_remaining_O2'].mean():.2f} % "
          f"(std = {late_df['Fe-terminal_remaining_O2'].std():.2f})")

    print(f"\n总用时：{time.time() - start_time:.2f} 秒")
