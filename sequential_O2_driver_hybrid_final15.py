#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequential_O2_driver_hybrid_final15.py

Sequential O2 pulse-dosing workflow with residual-O2-aware MD extensions,
hybrid preferred-height insertion with random below-wall fallback, optional one-time incoming velocity direction,
and a final long relaxation after the target O2 exposure is reached.

Workflow per pulse
------------------
1. Analyze residual free O2 before dosing.
2. Insert O2 molecules with a hybrid rule: first try the preferred local-surface
   height, then fall back to a random gas-region position below the Fermi wall
   if the preferred height is crowded/unavailable. Existing residual O2 is kept
   and included in overlap checks.
3. Run main MD for --md-steps.
4. Analyze residual free O2.
   - If residual O2 <= threshold: proceed to next pulse.
   - If residual O2 > threshold: run extension MD chunks of --extension-steps.
       * Continue while residual O2 decreases by at least --min-residual-decrease.
       * If residual O2 no longer decreases, proceed to next pulse, keeping residual O2.
5. After the target exposure is reached, run --final-md-steps additional MD steps
   (default 15000 = 15 ps when timestep = 1 fs).
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Sequential O2 dosing with residual-aware MD extensions and final 15 ps relaxation")
    p.add_argument("--initial", required=True, help="Initial clean slab structure")
    p.add_argument("--total-o2", type=int, default=104, help="Total O2 molecule exposure dose")
    p.add_argument("--o2-per-round", type=int, default=8, help="O2 molecules per exposure pulse")
    p.add_argument("--md-steps", type=int, default=3000, help="MD steps after each new O2 pulse")
    p.add_argument("--extension-steps", type=int, default=1000, help="Extra MD steps when residual O2 remains reactive")
    p.add_argument("--final-md-steps", type=int, default=15000, help="Final extra MD steps after target O2 exposure; 15000 = 15 ps for 1 fs timestep")
    p.add_argument("--residual-threshold", type=int, default=1, help="If residual free O2 <= threshold, add next pulse")
    p.add_argument("--min-residual-decrease", type=int, default=1, help="Continue extension MD only if residual free O2 decreases by at least this value")
    p.add_argument("--max-extensions", type=int, default=20, help="Safety limit for extension chunks per pulse")
    p.add_argument("--temperature", type=float, default=1500.0)
    p.add_argument("--workdir", default="sequential_O2_104_final15")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--add-script", default=None, help="Path to add_O2_hybrid_surface_then_wall.py")
    p.add_argument("--md-script", default=None, help="Path to run_MD_constrain_continue_batch.py")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--resume", action="store_true")

    # Insertion options
    p.add_argument("--surface-clearance", type=float, default=1.50, help="Lowest atom in new O2 must be at least this far above local connected surface, Å")
    p.add_argument("--local-radius", type=float, default=4.0, help="Local surface search radius, Å")
    p.add_argument("--bottom-zfrac", type=float, default=0.06)
    p.add_argument("--upper-zfrac", type=float, default=0.65, help="Fermi-wall center position as fractional z for insertion")
    p.add_argument("--wall-clearance", type=float, default=2.0, help="Highest atom in new O2 must be this far below Fermi wall, Å")
    p.add_argument("--preferred-height", type=float, default=2.50, help="First try to place O2 COM at local surface + this height, Å")
    p.add_argument("--preferred-attempts-per-o2", type=int, default=50000, help="Attempts at preferred height before random below-wall fallback")
    p.add_argument("--min-dist", type=float, default=1.78, help="Initial insertion overlap threshold, Å")
    p.add_argument("--hard-min-dist", type=float, default=1.55, help="Lowest allowed overlap threshold if insertion is crowded, Å")
    p.add_argument("--min-dist-step", type=float, default=0.05)
    p.add_argument("--relax-after-attempts", type=int, default=30000)
    p.add_argument("--bond", type=float, default=1.205)
    p.add_argument("--connect-scale", type=float, default=1.28)
    p.add_argument("--mo-bond-cut", type=float, default=2.45)
    p.add_argument("--free-o2-min", type=float, default=0.90)
    p.add_argument("--free-o2-max", type=float, default=1.45)
    p.add_argument("--direct-o2-to-surface", action="store_true", help="Mirror newly inserted O2 COM z-velocity downward at insertion only")

    # MD options inherited from previous batch script
    p.add_argument("--model-dir", default="MACE_models", help="Directory containing the MACE model files")
    p.add_argument("--model1", default="FeSiO_com1_run-123_stagetwo.model")
    p.add_argument("--model2", default="FeSiO_com2_run-456_stagetwo.model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--target_collect", type=int, default=80)
    p.add_argument("--energy_diff_threshold", type=float, default=1.00)
    p.add_argument("--force_diff_threshold", type=float, default=10.0)
    p.add_argument("--cooldown_steps", type=int, default=20)
    p.add_argument("--fix-bottom-zfrac", type=float, default=0.06)
    p.add_argument("--wall-A", type=float, default=5.0)
    p.add_argument("--wall-steepness", type=float, default=60.0)
    p.add_argument("--wall-upper-center-frac", type=float, default=0.65, help="Fermi-wall center position used by MD")
    return p.parse_args()


def abs_path(path):
    return str(Path(path).expanduser().resolve())


def run_cmd(cmd, cwd=None, log_file=None):
    print("\n[RUN] " + " ".join(map(str, cmd)))
    if cwd:
        print(f"[CWD] {cwd}")
    if log_file:
        with open(log_file, "w", buffering=1) as f:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    else:
        subprocess.check_call(cmd, cwd=cwd)


def analyze_free_o2(args, add_script, structure, out_json, cwd=None):
    cmd = [
        args.python, add_script,
        "-i", abs_path(structure),
        "--analyze-only",
        "--json-output", str(out_json),
        "--bottom-zfrac", str(args.bottom_zfrac),
        "--connect-scale", str(args.connect_scale),
        "--local-radius", str(args.local_radius),
        "--mo-bond-cut", str(args.mo_bond_cut),
        "--free-o2-min", str(args.free_o2_min),
        "--free-o2-max", str(args.free_o2_max),
    ]
    run_cmd(cmd, cwd=cwd, log_file=None)
    with open(out_json) as f:
        data = json.load(f)
    return int(data["n_free_o2"]), data


def build_add_cmd(args, add_script, current_input, dosed_input, add_now, round_seed):
    cmd = [
        args.python, add_script,
        "-i", abs_path(current_input),
        "-o", str(dosed_input),
        "--n-o2", str(add_now),
        "--bond", str(args.bond),
        "--min-dist", str(args.min_dist),
        "--hard-min-dist", str(args.hard_min_dist),
        "--min-dist-step", str(args.min_dist_step),
        "--relax-after-attempts", str(args.relax_after_attempts),
        "--temperature", str(args.temperature),
        "--seed", str(round_seed),
        "--bottom-zfrac", str(args.bottom_zfrac),
        "--connect-scale", str(args.connect_scale),
        "--local-radius", str(args.local_radius),
        "--surface-clearance", str(args.surface_clearance),
        "--upper-zfrac", str(args.upper_zfrac),
        "--wall-clearance", str(args.wall_clearance),
        "--preferred-height", str(args.preferred_height),
        "--preferred-attempts-per-o2", str(args.preferred_attempts_per_o2),
        "--mo-bond-cut", str(args.mo_bond_cut),
        "--free-o2-min", str(args.free_o2_min),
        "--free-o2-max", str(args.free_o2_max),
    ]
    if args.direct_o2_to_surface:
        cmd.append("--direct-o2-to-surface")
    return cmd


def build_md_cmd(args, md_script, input_structure, md_steps, traj_file, last_config):
    return [
        args.python, md_script,
        "--input", abs_path(input_structure),
        "--max-steps", str(md_steps),
        "--temperature", str(args.temperature),
        "--model-dir", args.model_dir,
        "--model1", args.model1,
        "--model2", args.model2,
        "--device", args.device,
        "--dtype", args.dtype,
        "--target_collect", str(args.target_collect),
        "--energy_diff_threshold", str(args.energy_diff_threshold),
        "--force_diff_threshold", str(args.force_diff_threshold),
        "--cooldown_steps", str(args.cooldown_steps),
        "--fix-bottom-zfrac", str(args.fix_bottom_zfrac),
        "--wall-A", str(args.wall_A),
        "--wall-steepness", str(args.wall_steepness),
        "--wall-upper-center-frac", str(args.wall_upper_center_frac),
        "--traj-file", traj_file,
        "--uncertain-file", "uncertain_structures.extxyz",
        "--last-config", last_config,
    ]


def write_summary_header(summary_path, resume):
    if not summary_path.exists() or not resume:
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "stage", "o2_added_this_round", "cumulative_o2_exposure",
                "input_before_dosing", "input_after_dosing", "final_after_md",
                "md_steps_this_stage", "residual_free_o2", "previous_residual_free_o2",
                "residual_decrease", "decision", "seed"
            ])


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    add_script = abs_path(args.add_script) if args.add_script else str(script_dir / "add_O2_hybrid_surface_then_wall.py")
    md_script = abs_path(args.md_script) if args.md_script else str(script_dir / "run_MD_constrain_continue_batch.py")

    for f in [add_script, md_script, args.initial]:
        if not Path(f).exists():
            raise FileNotFoundError(f"Cannot find required file: {f}")

    if abs(args.upper_zfrac - args.wall_upper_center_frac) > 1e-8:
        print(f"[WARN] upper-zfrac for insertion ({args.upper_zfrac}) differs from MD wall-upper-center-frac ({args.wall_upper_center_frac}).")

    n_rounds = math.ceil(args.total_o2 / args.o2_per_round)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(add_script, workdir / Path(add_script).name)
    shutil.copy2(md_script, workdir / Path(md_script).name)

    summary_path = workdir / "sequential_O2_final15_summary.csv"
    write_summary_header(summary_path, args.resume)

    current_input = abs_path(args.initial)
    cumulative = 0

    final_dir = workdir / "final_relaxation_15ps"
    final_done = final_dir / "final_15ps_final.extxyz"

    # Simple resume: find latest round final and reconstruct cumulative exposure.
    if args.resume:
        latest_final = None
        latest_round = 0
        for r in range(1, n_rounds + 1):
            round_dir = workdir / f"round_{r:02d}"
            candidate = round_dir / "final_for_next_round.extxyz"
            if candidate.exists() and candidate.stat().st_size > 0:
                latest_final = abs_path(candidate)
                latest_round = r
        if latest_final is not None:
            current_input = latest_final
            cumulative = min(latest_round * args.o2_per_round, args.total_o2)
            print(f"[RESUME] Completed round {latest_round}; continuing from {current_input}; cumulative exposure={cumulative}")
        if cumulative >= args.total_o2 and final_done.exists() and final_done.stat().st_size > 0:
            print(f"[RESUME] Target exposure and final relaxation already complete: {final_done}")
            return

    for r in range(1, n_rounds + 1):
        if args.resume and r <= cumulative // args.o2_per_round:
            continue

        remaining = args.total_o2 - cumulative
        add_now = min(args.o2_per_round, remaining)
        if add_now <= 0:
            break

        round_dir = workdir / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_seed = args.seed + r - 1
        dosed_input = round_dir / f"round_{r:02d}_plus_{add_now}O2.extxyz"

        print("\n" + "=" * 120)
        print(f"Round {r}/{n_rounds}: add {add_now} O2 | cumulative exposure after pulse = {cumulative + add_now}/{args.total_o2}")
        print("Insertion: preferred local surface height first; random below-wall fallback if crowded; residual O2 kept and checked for overlap")
        print(f"After insertion: main MD {args.md_steps} steps; residual-aware extension {args.extension_steps} steps/chunk if needed")
        print("=" * 120)

        before_json = round_dir / "before_dosing_free_o2.json"
        n_free_before, _ = analyze_free_o2(args, add_script, current_input, before_json, cwd=str(round_dir))
        print(f"[INFO] Residual free O2 before dosing: {n_free_before}")

        add_cmd = build_add_cmd(args, add_script, current_input, dosed_input, add_now, round_seed)
        run_cmd(add_cmd, cwd=str(round_dir), log_file=round_dir / "add_O2_below_wall.log")

        main_final = round_dir / "stage_main_final.extxyz"
        md_cmd = build_md_cmd(
            args, md_script, dosed_input, args.md_steps,
            traj_file="stage_main.traj",
            last_config=str(main_final),
        )
        md_cmd += ["--seed", str(round_seed)]
        run_cmd(md_cmd, cwd=str(round_dir), log_file=round_dir / "stage_main_md.log")

        if not main_final.exists() or main_final.stat().st_size == 0:
            raise RuntimeError(f"Main MD finished but final config not found: {main_final}")

        cumulative += add_now
        current_final = main_final
        residual_json = round_dir / "stage_main_free_o2.json"
        residual, _ = analyze_free_o2(args, add_script, current_final, residual_json, cwd=str(round_dir))

        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([
                r, "main", add_now, cumulative, current_input, abs_path(dosed_input), abs_path(current_final),
                args.md_steps, residual, "", "", "analyze_after_main_md", round_seed
            ])

        ext = 0
        while residual > args.residual_threshold and ext < args.max_extensions:
            ext += 1
            previous_residual = residual
            ext_final = round_dir / f"stage_ext_{ext:02d}_final.extxyz"
            ext_cmd = build_md_cmd(
                args, md_script, current_final, args.extension_steps,
                traj_file=f"stage_ext_{ext:02d}.traj",
                last_config=str(ext_final),
            )
            ext_cmd += ["--seed", str(round_seed + 1000 + ext)]
            print(
                f"[WAIT] Residual free O2 = {previous_residual} > {args.residual_threshold}; "
                f"running {args.extension_steps} extra MD steps (extension {ext})."
            )
            run_cmd(ext_cmd, cwd=str(round_dir), log_file=round_dir / f"stage_ext_{ext:02d}_md.log")

            if not ext_final.exists() or ext_final.stat().st_size == 0:
                raise RuntimeError(f"Extension MD finished but final config not found: {ext_final}")

            ext_json = round_dir / f"stage_ext_{ext:02d}_free_o2.json"
            new_residual, _ = analyze_free_o2(args, add_script, ext_final, ext_json, cwd=str(round_dir))
            decrease = previous_residual - new_residual
            current_final = ext_final
            residual = new_residual

            if residual <= args.residual_threshold:
                decision = "residual_le_threshold_add_next_pulse"
            elif decrease >= args.min_residual_decrease:
                decision = "residual_decreased_continue_waiting"
            else:
                decision = "residual_no_longer_decreases_add_next_pulse"

            with open(summary_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    r, f"extension_{ext:02d}", 0, cumulative, "", "", abs_path(current_final),
                    args.extension_steps, residual, previous_residual, decrease, decision, round_seed + 1000 + ext
                ])

            print(
                f"[CHECK] residual O2: {previous_residual} -> {residual} "
                f"(decrease={decrease}); decision={decision}"
            )

            if residual <= args.residual_threshold:
                break
            if decrease >= args.min_residual_decrease:
                continue
            break

        if ext >= args.max_extensions and residual > args.residual_threshold:
            print(
                f"[WARN] Reached max extensions ({args.max_extensions}) with residual free O2={residual}; "
                "proceeding to next dose to avoid infinite waiting."
            )

        final_for_next = round_dir / "final_for_next_round.extxyz"
        shutil.copy2(current_final, final_for_next)
        current_input = abs_path(final_for_next)

        print(
            f"[ROUND DONE] Round {r}: final residual free O2={residual}; "
            f"next input={current_input}"
        )

    # Final 15 ps relaxation after total exposure target has been inserted and processed.
    if args.final_md_steps > 0:
        final_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and final_done.exists() and final_done.stat().st_size > 0:
            print(f"[RESUME] Final relaxation already exists: {final_done}")
            current_input = abs_path(final_done)
        else:
            print("\n" + "=" * 120)
            print(f"Final relaxation after total O2 exposure: {args.final_md_steps} MD steps")
            print(f"Input structure: {current_input}")
            print("=" * 120)
            final_cmd = build_md_cmd(
                args, md_script, current_input, args.final_md_steps,
                traj_file="final_15ps.traj",
                last_config=str(final_done),
            )
            final_cmd += ["--seed", str(args.seed + 900000)]
            run_cmd(final_cmd, cwd=str(final_dir), log_file=final_dir / "final_15ps_md.log")
            if not final_done.exists() or final_done.stat().st_size == 0:
                raise RuntimeError(f"Final relaxation finished but final config not found: {final_done}")
            final_json = final_dir / "final_15ps_free_o2.json"
            final_residual, _ = analyze_free_o2(args, add_script, final_done, final_json, cwd=str(final_dir))
            with open(summary_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    "final", "final_relaxation", 0, cumulative, current_input, "", abs_path(final_done),
                    args.final_md_steps, final_residual, "", "", "final_relaxation_after_total_exposure", args.seed + 900000
                ])
            current_input = abs_path(final_done)

    print("\n" + "=" * 120)
    print("Sequential O2 workflow finished")
    print(f"Total O2 exposure dosed: {cumulative}/{args.total_o2} O2 molecules")
    print(f"Final structure: {current_input}")
    print(f"Summary CSV: {summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
