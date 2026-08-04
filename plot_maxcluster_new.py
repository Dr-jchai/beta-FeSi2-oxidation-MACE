#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取四个 CSV 文件，绘制四条曲线随时间（ps）的变化
- 横轴：时间 (ps)
- 纵轴：四个文件的第二列数据（归一化后）
- 新增：后期（最后7000步）平均值与标准差统计
- 输出：maxcluster.dat 和 maxcluster.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.signal import savgol_filter

# ----------------------- 配置部分 -----------------------
# 四个 CSV 文件路径
FILES = {
    "Fe_O_Fe_Fe_terminal": "Fe-terminal-Fe_O_Fe.csv",
    "Fe_O_Fe_Si_terminal": "Si-terminal-Fe_O_Fe.csv",
    "Si_O_Si_Fe_terminal": "Fe-terminal-Si_O_Si.csv",
    "Si_O_Si_Si_terminal": "Si-terminal-Si_O_Si.csv"
}

DT_FS = 1.0                    # 每步时间步长 (fs)
SMOOTH_WINDOW = 401            # Savitzky-Golay 平滑窗口大小
SMOOTH_ORDER = 3               # 多项式阶数

# 归一化因子
NORM_FACTOR = {
    "Fe_O_Fe": 64,
    "Si_O_Si": 64
}

# 颜色设置（与之前风格保持一致）
COLORS = [
    '#1E90FF',   # Si-O-Si (Si-terminal) - 天蓝色
    '#32CD32',   # Si-O-Si (Fe-terminal) - 绿色
    '#C41E3A',   # Fe-O-Fe (Fe-terminal) - 深酒红
    '#FF8C00'    # Fe-O-Fe (Si-terminal) - 橙色
]

# 图例标签
LABELS = [
    "Si-O-Si (Si-terminal)",
    "Si-O-Si (Fe-terminal)",
    "Fe-O-Fe (Fe-terminal)",
    "Fe-O-Fe (Si-terminal)"
]

# ----------------------- 读取数据 -----------------------
df_time = pd.read_csv(FILES["Fe_O_Fe_Fe_terminal"], sep=',', usecols=[0])
df_time.columns = ['step']
df_time['time_ps'] = df_time['step'] * DT_FS / 1000.0

df = pd.DataFrame()
df['step'] = df_time['step']
df['time_ps'] = df_time['time_ps']

data_cols = [
    ("Si_O_Si_Si_terminal", "Si_O_Si", FILES["Si_O_Si_Si_terminal"]),
    ("Si_O_Si_Fe_terminal", "Si_O_Si", FILES["Si_O_Si_Fe_terminal"]),
    ("Fe_O_Fe_Fe_terminal", "Fe_O_Fe", FILES["Fe_O_Fe_Fe_terminal"]),
    ("Fe_O_Fe_Si_terminal", "Fe_O_Fe", FILES["Fe_O_Fe_Si_terminal"])
]

for key, group, filepath in data_cols:
    temp_df = pd.read_csv(filepath, sep=',', usecols=[1])
    temp_df.columns = [key]
    norm = NORM_FACTOR[group]
    df[key] = temp_df[key] / norm

print(f"数据行数：{len(df)}")
print(f"时间范围：{df['time_ps'].min():.1f} ps → {df['time_ps'].max():.1f} ps")

# ----------------------- 后期统计（最后7000步） -----------------------
late_df = df[df['step'] >= df['step'].max() - 7999]   # 最后7000步

print("\n后期统计（最后7000步）：")
for key, label in zip(
    ["Si_O_Si_Si_terminal", "Si_O_Si_Fe_terminal",
     "Fe_O_Fe_Fe_terminal", "Fe_O_Fe_Si_terminal"],
    LABELS
):
    if key in late_df.columns:
        mean_val = late_df[key].mean()
        std_val = late_df[key].std()
        print(f"{label:25s} 平均 = {mean_val:.4f} 标准差 = {std_val:.4f}")

# ----------------------- 输出 maxcluster.dat -----------------------
output_df = df[['step', 'Si_O_Si_Si_terminal', 'Si_O_Si_Fe_terminal',
                'Fe_O_Fe_Fe_terminal', 'Fe_O_Fe_Si_terminal']].copy()
output_df.columns = ['step', 'Si_O_Si_Si', 'Si_O_Si_Fe', 'Fe_O_Fe_Fe', 'Fe_O_Fe_Si']
output_df.to_csv("maxcluster.dat", sep='\t', index=False, float_format='%.8f')
print("\n已输出 maxcluster.dat")

# ----------------------- 绘图 -----------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=1200)

plot_keys = ["Si_O_Si_Si_terminal", "Si_O_Si_Fe_terminal",
             "Fe_O_Fe_Fe_terminal", "Fe_O_Fe_Si_terminal"]

for i, (key, label, color) in enumerate(zip(plot_keys, LABELS, COLORS)):
    # 原始数据（细线 + 半透明）
    ax.plot(
        df['time_ps'],
        df[key],
        color=color,
        lw=0.9,
        alpha=0.5,
        marker='.',
        markersize=0.6,
        markeredgewidth=0
    )
    
    # Savitzky-Golay 平滑曲线
    if len(df) >= SMOOTH_WINDOW:
        smoothed = savgol_filter(df[key], window_length=SMOOTH_WINDOW, polyorder=SMOOTH_ORDER)
    else:
        smoothed = df[key].values
        
    ax.plot(
        df['time_ps'],
        smoothed,
        color=color,
        lw=3.4,
        linestyle='-',
        label=label
    )

# 设置坐标轴范围
x_min = df['time_ps'].min() - 0.5
x_max = df['time_ps'].max() + 0.5
y_max = df[plot_keys].max().max() * 1.1

ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.1, y_max)

# 主次刻度
ax.xaxis.set_major_locator(MultipleLocator(5.0))
ax.xaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# 刻度样式
ax.tick_params(axis='both', which='major', direction='out', length=8, width=1.2)
ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1.0)

# 加粗边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 标签和标题
ax.set_xlabel("Time (ps)", fontsize=15)
ax.set_ylabel("Normalized cluster size", fontsize=14)
ax.set_title("Evolution of largest bridged clusters\n(Si-O-Si and Fe-O-Fe)", 
             fontsize=15, pad=15)

# 图例
ax.legend(fontsize=15, ncol=2, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig("maxcluster.png", dpi=1200, bbox_inches='tight')
print("绘图已保存为：maxcluster.png")
