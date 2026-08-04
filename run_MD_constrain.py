#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin  # 使用 Langevin 恒温器
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase import units
from mace.calculators import MACECalculator

# 强制实时打印（slurm 必备）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = open("error.out", "a", buffering=1)

# ============================== 参数解析 ==============================
parser = argparse.ArgumentParser(description="Run MACE MD with committee average forces")
parser.add_argument("--target_collect", type=int, default=80)
parser.add_argument("--energy_diff_threshold", type=float, default=1.00)
parser.add_argument("--force_diff_threshold", type=float, default=10.0,
    help="Max component-wise force disagreement threshold in eV/Å (default: 1.5)")
parser.add_argument("--cooldown_steps", type=int, default=20)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--seed", type=int, default=None,
    help="Random seed for reproducibility (fixes numpy/random, not GPU non-determinism)")
args = parser.parse_args()

TARGET_COLLECT = args.target_collect
ENERGY_DIFF_THRESHOLD = args.energy_diff_threshold
FORCE_DIFF_THRESHOLD = args.force_diff_threshold
COOLDOWN_STEPS = args.cooldown_steps
START_STEP_OFFSET = args.start_step

# ============================== 随机种子 ==============================
if args.seed is not None:
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"[INFO] 随机种子已设置：seed = {args.seed}")
else:
    print("[INFO] 未设置随机种子，每次运行结果不同")

# ============================== 配置 ==============================
MODEL_DIR = os.environ.get("MACE_MODEL_DIR", "MACE_models")
model1_path = f"{MODEL_DIR}/FeSiO_com1_run-123_stagetwo.model"
model2_path = f"{MODEL_DIR}/FeSiO_com2_run-456_stagetwo.model"

UNCERTAIN_FILE = "uncertain_structures.extxyz"
TRAJ_FILE = "nvt_1200K.traj"
LAST_CONFIG = "last_config_with_velocities.extxyz"
INIT_XYZ = os.environ.get("FESI2_INIT_XYZ", "FeSi2.extxyz")

if os.path.exists(UNCERTAIN_FILE):
    os.remove(UNCERTAIN_FILE)
    print(f"[INFO] 已删除旧的 {UNCERTAIN_FILE}，本轮重新采集")

print("[INFO] 正在加载单个 MACE 模型用于不确定性计算...")
model1_calc = MACECalculator(model_paths=model1_path, device="cuda", default_dtype="float32")
model2_calc = MACECalculator(model_paths=model2_path, device="cuda", default_dtype="float32")

print("[INFO] 正在加载 MACE committee（用于平均力跑 MD）...")
committee_calc = MACECalculator(
    model_paths=[model1_path, model2_path],
    device="cuda",
    default_dtype="float32",
)

# ============================== 物理正确的上墙 Calculator ==============================
class UpperWallCalculator(Calculator):
    """
    物理正确上墙：
    z_frac < upper_center : V ≈ 0
    z_frac > upper_center : V → A
    力向下推回（负方向）
    """
    implemented_properties = ["energy", "forces"]
    def __init__(self, A=5.0, steepness=60.0, upper_center_frac=0.6, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.steepness = steepness
        self.upper_center_frac = upper_center_frac

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        scaled = atoms.get_scaled_positions(wrap=False)
        z_frac = scaled[:, 2]
        cell_z = atoms.cell[2, 2]
        exp_upper = np.exp(self.steepness * (self.upper_center_frac - z_frac))
        v_upper = self.A / (1.0 + exp_upper)
        fz_upper_frac = - self.A * self.steepness * exp_upper / (1.0 + exp_upper)**2
        forces = np.zeros_like(atoms.positions)
        forces[:, 2] = fz_upper_frac / cell_z
        self.results["energy"] = np.sum(v_upper)
        self.results["forces"] = forces

# ============================== 初始结构 ==============================
if os.path.exists(LAST_CONFIG) and os.path.getsize(LAST_CONFIG) > 100:
    print("[INFO] 检测到有效上一轮构型，继续无缝 MD 采样...")
    atoms = read(LAST_CONFIG)
else:
    print("[INFO] 无有效历史构型，从初始结构开始...")
    atoms = read(INIT_XYZ, index=0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1200.0)

# 固定底层原子
scaled_pos = atoms.get_scaled_positions()
mask_bottom = scaled_pos[:, 2] <= 0.06
print(f"[INFO] 固定底层原子：z_frac ≤ 0.06，共 {mask_bottom.sum()} 个原子（总 {len(atoms)} 个）")
atoms.set_constraint(FixAtoms(mask=mask_bottom))

# 清零固定原子速度
velocities = atoms.get_velocities()
if velocities is not None:
    velocities[mask_bottom] = 0.0
    atoms.set_velocities(velocities)
print("[INFO] 已清零固定底层原子的速度")

# 叠加上墙（MD 使用 committee 平均 + wall）
wall_calc = UpperWallCalculator(A=5.0, steepness=60.0, upper_center_frac=0.65)
atoms.calc = SumCalculator([committee_calc, wall_calc])

# ============================== MD 设置 ==============================
dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=1500.0,
    friction=0.1,
)

traj = Trajectory(TRAJ_FILE, "a", atoms)
last_collected_step = START_STEP_OFFSET - 99999
last_collected_atoms = None
collected_count = 0

def clean_mace_info(at):
    keys_to_del = []
    for key, val in at.info.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            keys_to_del.append(key)
    for k in set(keys_to_del):
        del at.info[k]

def log_and_save():
    global last_collected_step, last_collected_atoms, collected_count
    local_step = dyn.nsteps
    step = dyn.nsteps + START_STEP_OFFSET
    temp = atoms.get_temperature()
    epot_mean = atoms.get_potential_energy()  # committee avg + wall

    # 计算两个模型的单独能量和力（不含 wall，wall 对两者相同，不影响 disagreement）
    current_calc = atoms.calc
    atoms.calc = model1_calc
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    atoms.calc = model2_calc
    e2 = atoms.get_potential_energy()
    f2 = atoms.get_forces()
    atoms.calc = current_calc  # 恢复

    diff_per_atom = abs(e1 - e2) / len(atoms)
    delta_meV = diff_per_atom * 1000.0
    max_force_diff = np.max(np.abs(f1 - f2))

    trigger = (
        local_step > 0
        and (diff_per_atom > ENERGY_DIFF_THRESHOLD or max_force_diff > FORCE_DIFF_THRESHOLD)
        and (step - last_collected_step) > COOLDOWN_STEPS
    )

    mark = " <<--- COLLECTED!" if trigger else ""
    z_frac = atoms.get_scaled_positions(wrap=False)[:, 2]
    print(
        f"Step {step:6d} (local {local_step:4d}) | T = {temp:7.1f} K | "
        f"Epot_mean = {epot_mean:10.6f} eV | "
        f"ΔE = {delta_meV:6.2f} meV/atom | "
        f"ΔF_max = {max_force_diff:6.3f} eV/Å{mark} | "
        f"z_frac: {z_frac.min():.3f}~{z_frac.max():.3f}"
    )

    if trigger:
        at_save = atoms.copy()
        clean_mace_info(at_save)
        at_save.info.update(
            {
                "step": step,
                "time_fs": step * 1.0,
                "temperature_K": temp,
                "energy_diff_meV_per_atom": delta_meV,
                "max_force_diff_eV_per_A": float(max_force_diff),
                "E_model_1": float(e1),
                "E_model_2": float(e2),
                "E_mean": float(epot_mean),
                "source": "mace_committee_2models",
                "config_type": "high_uncertainty",
            }
        )
        at_save.calc = None
        write(UNCERTAIN_FILE, at_save, format="extxyz", append=True)
        last_collected_step = step
        last_collected_atoms = at_save.copy()
        collected_count += 1

    if local_step > 0:
        traj.write(atoms)

    if collected_count >= TARGET_COLLECT:
        print("\n" + "=" * 80)
        print(f"本轮 MD 结束！成功采集 {collected_count} 个高不确定性结构")
        print(f"最后采集步数: {last_collected_step}")
        print("=" * 80)
        if last_collected_atoms is not None:
            write(LAST_CONFIG, last_collected_atoms, format="extxyz")
            print(f"last_config 已保存")
        traj.close()
        os._exit(0)

# 初始计算
print("[INFO] 初始构型：进行一次 committee + wall 计算...")
log_and_save()

dyn.attach(log_and_save, interval=1)

print("=" * 80)
print("MACE Committee Active Learning MD 已启动（Langevin 稳定版）")
print("→ 底层固定 + 速度清零")
print("→ 上墙物理正确")
print("→ 不确定性判断：ΔE_per_atom > {:.4f} eV 或 ΔF_max > {:.3f} eV/Å".format(ENERGY_DIFF_THRESHOLD, FORCE_DIFF_THRESHOLD))
print("→ Langevin 恒温（friction=0.1）")
print("=" * 80)

MAX_STEPS = 28000
print(f"开始运行 MD，计划步数：{MAX_STEPS}")
dyn.run(MAX_STEPS)
print("MD 运行结束")
