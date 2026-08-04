#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_MD_constrain_continue_batch.py

A patched version of run_MD_constrain_continue.py for sequential O2 dosing.
It keeps the original logic: MACE committee MD + upper Fermi-like wall + bottom
fixed atoms + uncertainty collection. Main changes:

1. Input structure is no longer hard-coded; use --input.
2. MAX_STEPS is no longer hard-coded; use --max-steps.
3. The final MD configuration is always saved to --last-config, even if no
   high-uncertainty structure is collected.
4. Each round can run in its own directory without accidentally appending to a
   trajectory from another round.

Example
-------
python run_MD_constrain_continue_batch.py --input input_with_8O2.extxyz \
    --max-steps 1000 --temperature 1500 --seed 2025
"""

import os
import sys
import argparse
import random
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase import units
from mace.calculators import MACECalculator

# Force real-time printing, useful for Slurm logs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = open("error.out", "a", buffering=1)


# ============================== Argument parsing ==============================
parser = argparse.ArgumentParser(description="Run MACE MD with committee average forces")
parser.add_argument("--input", default="input_with_8O2.extxyz", help="Initial structure for this MD round")
parser.add_argument("--model-dir", default="MACE_models", help="Directory containing the MACE model files")
parser.add_argument("--model1", default="FeSiO_com1_run-123_stagetwo.model")
parser.add_argument("--model2", default="FeSiO_com2_run-456_stagetwo.model")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", default="float32")
parser.add_argument("--temperature", type=float, default=1500.0)
parser.add_argument("--max-steps", type=int, default=13000)
parser.add_argument("--timestep-fs", type=float, default=1.0)
parser.add_argument("--friction", type=float, default=0.1)
parser.add_argument("--target_collect", type=int, default=80)
parser.add_argument("--energy_diff_threshold", type=float, default=1.00)
parser.add_argument(
    "--force_diff_threshold",
    type=float,
    default=10.0,
    help="Max component-wise force disagreement threshold in eV/Å",
)
parser.add_argument("--cooldown_steps", type=int, default=20)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--uncertain-file", default="uncertain_structures.extxyz")
parser.add_argument("--traj-file", default="nvt_1500K.traj")
parser.add_argument("--last-config", default="final_config_with_velocities.extxyz")
parser.add_argument("--continue-from-traj", action="store_true", help="Continue from the last frame of --traj-file if it exists")
parser.add_argument("--fix-bottom-zfrac", type=float, default=0.06)
parser.add_argument("--wall-A", type=float, default=5.0)
parser.add_argument("--wall-steepness", type=float, default=60.0)
parser.add_argument("--wall-upper-center-frac", type=float, default=0.65)
args = parser.parse_args()

TARGET_COLLECT = args.target_collect
ENERGY_DIFF_THRESHOLD = args.energy_diff_threshold
FORCE_DIFF_THRESHOLD = args.force_diff_threshold
COOLDOWN_STEPS = args.cooldown_steps
START_STEP_OFFSET = args.start_step
TARGET_TEMPERATURE_K = args.temperature

# ============================== Random seed ==============================
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"[INFO] Random seed set: seed = {args.seed}")
else:
    print("[INFO] No random seed set; run is stochastic")

# ============================== Configuration ==============================
model1_path = os.path.join(args.model_dir, args.model1)
model2_path = os.path.join(args.model_dir, args.model2)

UNCERTAIN_FILE = args.uncertain_file
TRAJ_FILE = args.traj_file
LAST_CONFIG = args.last_config
INIT_XYZ = args.input

if os.path.exists(UNCERTAIN_FILE):
    os.remove(UNCERTAIN_FILE)
    print(f"[INFO] Removed old {UNCERTAIN_FILE}; this round will collect anew")

print("[INFO] Loading individual MACE models for uncertainty evaluation...")
model1_calc = MACECalculator(model_paths=model1_path, device=args.device, default_dtype=args.dtype)
model2_calc = MACECalculator(model_paths=model2_path, device=args.device, default_dtype=args.dtype)

print("[INFO] Loading MACE committee for average-force MD...")
committee_calc = MACECalculator(
    model_paths=[model1_path, model2_path],
    device=args.device,
    default_dtype=args.dtype,
)


# ============================== Upper-wall calculator ==============================
class UpperWallCalculator(Calculator):
    """
    Smooth upper wall:
    z_frac < upper_center : V approx 0
    z_frac > upper_center : V approaches A
    Force pushes atoms downward.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, A=5.0, steepness=60.0, upper_center_frac=0.65, **kwargs):
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
        fz_upper_frac = -self.A * self.steepness * exp_upper / (1.0 + exp_upper) ** 2

        forces = np.zeros_like(atoms.positions)
        forces[:, 2] = fz_upper_frac / cell_z

        self.results["energy"] = float(np.sum(v_upper))
        self.results["forces"] = forces


# ============================== Initial structure ==============================
continued_from_traj = False
if args.continue_from_traj and os.path.exists(TRAJ_FILE) and os.path.getsize(TRAJ_FILE) > 0:
    print(f"[INFO] Continuing MD from last frame of {TRAJ_FILE} ...")
    traj_in = Trajectory(TRAJ_FILE)
    atoms = traj_in[-1].copy()
    old_nframes = len(traj_in)
    traj_in.close()
    continued_from_traj = True

    velocities = atoms.get_velocities()
    if velocities is None:
        raise RuntimeError(f"{TRAJ_FILE} last frame has no velocities; seamless continuation is impossible.")

    print("[INFO] Read last-frame positions and velocities successfully")
    print(f"[INFO] Old trajectory frames: {old_nframes}")
    print(f"[INFO] Initial instantaneous temperature: {atoms.get_temperature():.2f} K")

    if START_STEP_OFFSET == 0:
        START_STEP_OFFSET = old_nframes - 1
        print(f"[INFO] Auto START_STEP_OFFSET = {START_STEP_OFFSET}")
else:
    print(f"[INFO] Starting this MD round from input structure: {INIT_XYZ}")
    atoms = read(INIT_XYZ, index=0)
    velocities = atoms.get_velocities()
    if velocities is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TARGET_TEMPERATURE_K, force_temp=True)
        print(f"[INFO] Initialized velocities at {TARGET_TEMPERATURE_K:.1f} K")
    else:
        print(f"[INFO] Input contains velocities; T_initial = {atoms.get_temperature():.2f} K")

# ============================== Bottom constraints ==============================
scaled_pos = atoms.get_scaled_positions(wrap=False)
mask_bottom = scaled_pos[:, 2] <= args.fix_bottom_zfrac
print(
    f"[INFO] Fixing bottom atoms: z_frac <= {args.fix_bottom_zfrac:.3f}, "
    f"count = {mask_bottom.sum()} / {len(atoms)}"
)
atoms.set_constraint(FixAtoms(mask=mask_bottom))

velocities = atoms.get_velocities()
if velocities is not None:
    velocities[mask_bottom] = 0.0
    atoms.set_velocities(velocities)
print("[INFO] Bottom-atom velocities zeroed")

# ============================== Calculator stack ==============================
wall_calc = UpperWallCalculator(
    A=args.wall_A,
    steepness=args.wall_steepness,
    upper_center_frac=args.wall_upper_center_frac,
)
atoms.calc = SumCalculator([committee_calc, wall_calc])

# ============================== MD setup ==============================
dyn = Langevin(
    atoms,
    timestep=args.timestep_fs * units.fs,
    temperature_K=TARGET_TEMPERATURE_K,
    friction=args.friction,
)

traj_mode = "a" if continued_from_traj else "w"
traj = Trajectory(TRAJ_FILE, traj_mode, atoms)

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


def write_final_config(reason="normal_end"):
    final_atoms = atoms.copy()
    clean_mace_info(final_atoms)
    final_atoms.info.update(
        {
            "source": "mace_committee_md_final",
            "finish_reason": reason,
            "temperature_K": float(atoms.get_temperature()),
            "md_steps_local": int(dyn.nsteps),
            "md_steps_global": int(dyn.nsteps + START_STEP_OFFSET),
        }
    )
    final_atoms.calc = None
    write(LAST_CONFIG, final_atoms, format="extxyz")
    print(f"[INFO] Final MD configuration saved to {LAST_CONFIG}")


def finish_and_exit(reason="target_collect_reached"):
    global last_collected_atoms
    print("\n" + "=" * 80)
    print(f"MD round stopped: {reason}")
    print(f"Collected high-uncertainty structures: {collected_count}")
    print(f"Current global step: {dyn.nsteps + START_STEP_OFFSET}")
    print("=" * 80)

    if last_collected_atoms is not None:
        write("last_high_uncertainty_config.extxyz", last_collected_atoms, format="extxyz")
        print("[INFO] Last high-uncertainty config saved to last_high_uncertainty_config.extxyz")

    write_final_config(reason=reason)
    traj.close()
    raise SystemExit(0)


def log_and_save():
    global last_collected_step, last_collected_atoms, collected_count

    local_step = dyn.nsteps
    step = dyn.nsteps + START_STEP_OFFSET
    temp = atoms.get_temperature()
    epot_mean = atoms.get_potential_energy()  # committee avg + wall

    # Compute individual model energies and forces. Wall is identical and omitted from disagreement.
    current_calc = atoms.calc

    atoms.calc = model1_calc
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()

    atoms.calc = model2_calc
    e2 = atoms.get_potential_energy()
    f2 = atoms.get_forces()

    atoms.calc = current_calc

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
        f"Step {step:6d} (local {local_step:4d}) | "
        f"T = {temp:7.1f} K | "
        f"Epot_mean = {epot_mean:10.6f} eV | "
        f"ΔE = {delta_meV:8.2f} meV/atom | "
        f"ΔF_max = {max_force_diff:8.3f} eV/Å{mark} | "
        f"z_frac: {z_frac.min():.3f}~{z_frac.max():.3f}"
    )

    if trigger:
        at_save = atoms.copy()
        clean_mace_info(at_save)
        at_save.info.update(
            {
                "step": step,
                "time_fs": step * args.timestep_fs,
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
        finish_and_exit(reason="target_collect_reached")


# ============================== Initial calculation ==============================
print("[INFO] Initial configuration: committee + wall calculation...")
log_and_save()

dyn.attach(log_and_save, interval=1)

print("=" * 80)
print("MACE Committee Active-Learning MD started")
print(f"→ Input structure: {INIT_XYZ}")
print(f"→ Bottom fixed: z_frac <= {args.fix_bottom_zfrac:.3f}")
print(f"→ Upper wall: A={args.wall_A}, steepness={args.wall_steepness}, center={args.wall_upper_center_frac}")
print(
    "→ Uncertainty criterion: ΔE_per_atom > {:.4f} eV or ΔF_max > {:.3f} eV/Å".format(
        ENERGY_DIFF_THRESHOLD, FORCE_DIFF_THRESHOLD
    )
)
print(f"→ Langevin thermostat: friction={args.friction}, T={TARGET_TEMPERATURE_K:.1f} K")
print(f"→ Trajectory: {TRAJ_FILE} (mode={traj_mode})")
print(f"→ Final structure: {LAST_CONFIG}")
print("=" * 80)

print(f"[INFO] Running MD for max_steps = {args.max_steps}")
try:
    dyn.run(args.max_steps)
except SystemExit:
    raise
except Exception as exc:
    print(f"[ERROR] MD failed: {exc}")
    try:
        write_final_config(reason="exception_before_end")
    finally:
        traj.close()
    raise

print("[INFO] MD run completed normally")
write_final_config(reason="normal_end")
traj.close()
