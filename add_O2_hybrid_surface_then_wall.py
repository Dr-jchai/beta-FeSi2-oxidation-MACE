#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_O2_hybrid_surface_then_wall.py

Residual-aware O2 insertion for sequential FeSi2 oxidation MD.

Main behavior
-------------
1. Existing residual free O2 is NOT removed.
2. Free gas-phase O2 is identified and excluded from connected-surface detection.
3. The substrate/oxide surface is defined as atoms connected to bottom atoms.
4. New O2 molecules are inserted with a hybrid rule:
   first try center_z = local connected surface + preferred_height;
   if this is too crowded/unavailable, fall back to a random z in the gas/vacuum
   region between local connected surface + surface_clearance and
   Fermi wall position - wall_clearance.
5. Overlap checks include all existing atoms, including residual free O2, and all
   new O2 already inserted in the current pulse.
6. If insertion is difficult, the nonbonded minimum distance is gradually relaxed
   from --min-dist down to --hard-min-dist, never below --hard-min-dist.
7. Existing velocities are preserved. Newly inserted O atoms receive Maxwell-
   Boltzmann velocities. With --direct-o2-to-surface, the COM z-velocity of each
   newly inserted O2 is mirrored downward only once at insertion.
"""

import argparse
import json
import os
from pathlib import Path
from collections import deque

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.data import covalent_radii, atomic_numbers
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid O2 insertion: preferred local-surface height first, fallback random below wall")
    p.add_argument("-i", "--input", default="POSCAR", help="Input structure")
    p.add_argument("-o", "--output", default="input_plus_O2.extxyz", help="Output structure")
    p.add_argument("--n-o2", type=int, default=8, help="Number of O2 molecules to insert in this pulse")
    p.add_argument("--bond", type=float, default=1.205, help="O-O bond length, Å")
    p.add_argument("--min-dist", type=float, default=1.78, help="Initial minimum nonbonded distance to existing atoms, Å")
    p.add_argument("--hard-min-dist", type=float, default=1.55, help="Lowest allowed nonbonded distance if insertion is crowded, Å")
    p.add_argument("--min-dist-step", type=float, default=0.05, help="Step size for gradual min-distance relaxation, Å")
    p.add_argument("--relax-after-attempts", type=int, default=30000, help="Relax min-dist after this many failed attempts per O2")
    p.add_argument("--temperature", type=float, default=1500.0, help="Temperature for new-O velocities")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--max-attempts-per-o2", type=int, default=300000)

    # Surface detection / insertion geometry
    p.add_argument("--bottom-zfrac", type=float, default=0.06, help="Bottom anchor threshold in fractional z")
    p.add_argument("--connect-scale", type=float, default=1.28, help="Covalent-radius scale for connectivity graph")
    p.add_argument("--local-radius", type=float, default=4.0, help="Lateral radius for local surface height, Å")
    p.add_argument("--surface-clearance", type=float, default=1.50, help="Lowest atom in new O2 must be at least this far above local connected surface, Å")
    p.add_argument("--upper-zfrac", type=float, default=0.65, help="Fermi-wall center position as fractional z")
    p.add_argument("--wall-clearance", type=float, default=2.0, help="Highest atom in new O2 must be at least this far below Fermi-wall center, Å")
    p.add_argument("--preferred-height", type=float, default=2.50, help="Preferred O2 center height above local connected surface before fallback, Å")
    p.add_argument("--preferred-attempts-per-o2", type=int, default=50000, help="Try this many attempts at preferred height before random below-wall fallback")

    # Free O2 identification
    p.add_argument("--free-o2-min", type=float, default=0.90, help="Minimum O-O distance for free O2 identification, Å")
    p.add_argument("--free-o2-max", type=float, default=1.45, help="Maximum O-O distance for free O2 identification, Å")
    p.add_argument("--mo-bond-cut", type=float, default=2.45, help="If O is within this distance of non-O atom, it is not free gas O2")

    # Output / analysis / velocity
    p.add_argument("--analyze-only", action="store_true", help="Only count residual free O2 and surface; do not insert O2")
    p.add_argument("--json-output", default=None, help="Optional JSON output for residual free O2 analysis")
    p.add_argument("--keep-stationary", action="store_true", help="Remove COM translation after velocity assignment")
    p.add_argument("--no-velocities", action="store_true", help="Do not write velocities")
    p.add_argument("--direct-o2-to-surface", action="store_true", help="Mirror newly inserted O2 COM z-velocity downward at insertion only")
    return p.parse_args()


def detect_vasp_output(path: str) -> bool:
    name = Path(path).name.upper()
    suffix = Path(path).suffix.lower()
    return name in {"POSCAR", "CONTCAR"} or suffix in {".vasp", ".poscar"}


def write_structure(path: str, atoms: Atoms):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if detect_vasp_output(path):
        write(path, atoms, format="vasp", direct=True, vasp5=True, sort=False, ignore_constraints=True)
    else:
        write(path, atoms, format="extxyz")


def random_unit_vector(rng):
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return random_unit_vector(rng)
    return v / n


def mic_cart_from_frac_delta(df, cell, pbc=(True, True, False)):
    df = np.array(df, dtype=float)
    for k in range(3):
        if pbc[k]:
            df[..., k] -= np.rint(df[..., k])
    return df @ cell


def pair_distance(i, j, scaled, cell, pbc=(True, True, False)):
    dcart = mic_cart_from_frac_delta(scaled[j] - scaled[i], cell, pbc)
    return float(np.linalg.norm(dcart))


def min_distance_to_existing(new_positions, existing_positions, cell):
    """Minimum distance from new atoms to existing atoms, using x/y periodic MIC only."""
    if len(existing_positions) == 0:
        return np.inf
    inv_cell = np.linalg.inv(cell)
    new_frac = new_positions @ inv_cell
    exist_frac = existing_positions @ inv_cell
    min_d = np.inf
    for nf in new_frac:
        df = exist_frac - nf
        df[:, 0] -= np.rint(df[:, 0])
        df[:, 1] -= np.rint(df[:, 1])
        # slab/vacuum direction is non-periodic
        dcart = df @ cell
        d = np.sqrt(np.sum(dcart * dcart, axis=1))
        local = float(np.min(d))
        if local < min_d:
            min_d = local
    return min_d


def identify_free_o2(atoms, args):
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    scaled = atoms.get_scaled_positions(wrap=False)
    O_indices = [i for i, s in enumerate(symbols) if s == "O"]
    nonO_indices = [i for i, s in enumerate(symbols) if s != "O"]

    if len(O_indices) < 2:
        return [], set()

    bound_to_substrate = set()
    nonO_pos = positions[nonO_indices] if nonO_indices else np.empty((0, 3))
    for oi in O_indices:
        if len(nonO_pos) == 0:
            continue
        dmin = min_distance_to_existing(positions[[oi]], nonO_pos, cell)
        if dmin <= args.mo_bond_cut:
            bound_to_substrate.add(oi)

    candidate_O = [i for i in O_indices if i not in bound_to_substrate]
    used = set()
    pairs = []
    for idx, i in enumerate(candidate_O):
        if i in used:
            continue
        best_j = None
        best_d = 1e9
        for j in candidate_O[idx + 1:]:
            if j in used:
                continue
            d = pair_distance(i, j, scaled, cell, pbc=(True, True, False))
            if args.free_o2_min <= d <= args.free_o2_max and d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)

    return pairs, set(used)


def build_connected_component(atoms, exclude_indices, args):
    symbols = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.cell.array, dtype=float)
    scaled = atoms.get_scaled_positions(wrap=False)
    active = [i for i in range(len(atoms)) if i not in exclude_indices]
    if not active:
        raise RuntimeError("No non-gas atoms available for surface detection.")

    zfrac = scaled[:, 2]
    bottom = [i for i in active if zfrac[i] <= args.bottom_zfrac]
    if not bottom:
        zvals = np.array([zfrac[i] for i in active])
        thresh = float(np.quantile(zvals, 0.10))
        bottom = [i for i in active if zfrac[i] <= thresh]
        print(f"[WARN] No atoms below bottom-zfrac={args.bottom_zfrac:.3f}; fallback threshold={thresh:.3f}")

    adj = {i: [] for i in active}
    for a_pos, i in enumerate(active):
        zi = atomic_numbers.get(symbols[i], 0)
        ri = covalent_radii[zi] if zi > 0 else 1.2
        for j in active[a_pos + 1:]:
            zj = atomic_numbers.get(symbols[j], 0)
            rj = covalent_radii[zj] if zj > 0 else 1.2
            cutoff = args.connect_scale * (ri + rj)
            d = pair_distance(i, j, scaled, cell, pbc=(True, True, False))
            if d <= cutoff:
                adj[i].append(j)
                adj[j].append(i)

    seen = set(bottom)
    q = deque(bottom)
    while q:
        i = q.popleft()
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                q.append(j)

    if len(seen) < max(5, int(0.2 * len(active))):
        print(f"[WARN] Connected component is small: {len(seen)}/{len(active)} active atoms. Check connectivity settings.")
    return sorted(seen)


def lateral_distance_to_point(atom_positions, center_cart, cell):
    inv = np.linalg.inv(cell)
    af = atom_positions @ inv
    cf = center_cart @ inv
    df = af - cf
    df[:, 0] -= np.rint(df[:, 0])
    df[:, 1] -= np.rint(df[:, 1])
    df[:, 2] = 0.0
    dcart = df @ cell
    return np.sqrt(np.sum(dcart * dcart, axis=1))


def local_surface_z(connected_positions, center_xy_cart, cell, local_radius):
    dlat = lateral_distance_to_point(connected_positions, center_xy_cart, cell)
    mask = dlat <= local_radius
    if np.any(mask):
        return float(np.max(connected_positions[mask, 2]))
    return float(np.max(connected_positions[:, 2]))


def analyze_structure(atoms, args):
    free_pairs, free_O = identify_free_o2(atoms, args)
    connected = build_connected_component(atoms, free_O, args)
    connected_positions = atoms.positions[connected]
    return {
        "n_atoms": int(len(atoms)),
        "n_free_o2": int(len(free_pairs)),
        "n_free_o_atoms": int(len(free_O)),
        "free_o2_pairs": [(int(i), int(j)) for i, j in free_pairs],
        "n_connected_atoms": int(len(connected)),
        "global_connected_surface_z_A": float(np.max(connected_positions[:, 2])),
        "global_connected_bottom_z_A": float(np.min(connected_positions[:, 2])),
    }, free_pairs, free_O, connected


def direct_new_o2_com_velocity_toward_surface(velocities, start_index, n_o2):
    """Mirror only the COM z velocity of newly inserted O2 molecules downward once."""
    v = np.array(velocities, dtype=float, copy=True)
    flipped = 0
    for k in range(n_o2):
        i = start_index + 2 * k
        j = i + 1
        vcom_z = 0.5 * (v[i, 2] + v[j, 2])
        if vcom_z > 0.0:
            v[i, 2] -= 2.0 * vcom_z
            v[j, 2] -= 2.0 * vcom_z
            flipped += 1
    return v, flipped


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.hard_min_dist > args.min_dist:
        raise ValueError("--hard-min-dist must be <= --min-dist")
    if args.hard_min_dist < 1.20:
        raise ValueError("--hard-min-dist is too small; use a safer value such as 1.50-1.60 Å")

    atoms = read(args.input, index=0)
    atoms.pbc = (True, True, False)
    cell = np.asarray(atoms.cell.array, dtype=float)
    if abs(np.linalg.det(cell)) < 1e-8:
        raise RuntimeError("Input cell is singular or missing.")

    analysis, free_pairs, free_O, connected = analyze_structure(atoms, args)
    if args.analyze_only:
        print(json.dumps(analysis, indent=2))
        if args.json_output:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
            with open(args.json_output, "w") as f:
                json.dump(analysis, f, indent=2)
        return

    connected_positions = atoms.positions[connected]
    global_surface_z = analysis["global_connected_surface_z_A"]
    wall_z = float(args.upper_zfrac * cell[2, 2])
    max_atom_z = wall_z - args.wall_clearance

    print("=" * 100)
    print("Random below-wall O2 insertion WITHOUT purging residual O2")
    print(f"Input                    : {args.input}")
    print(f"Output                   : {args.output}")
    print(f"Residual free O2 before   : {analysis['n_free_o2']} molecules, kept in the box")
    print(f"O2 pulse size             : {args.n_o2}")
    print(f"Insertion z mode          : hybrid: preferred local_surface_z + {args.preferred_height:.3f} Å, then random below-wall fallback")
    print(f"Surface clearance         : lowest new O atom >= local_surface_z + {args.surface_clearance:.3f} Å")
    print(f"Fermi wall z              : {wall_z:.3f} Å")
    print(f"Highest new O atom <=     : {max_atom_z:.3f} Å  (wall clearance {args.wall_clearance:.3f} Å)")
    print(f"Initial min distance      : {args.min_dist:.3f} Å")
    print(f"Hard min distance         : {args.hard_min_dist:.3f} Å")
    print(f"Connected atoms           : {analysis['n_connected_atoms']} / {len(atoms)}")
    print(f"Global connected surface z: {global_surface_z:.3f} Å")
    print(f"Seed                      : {args.seed}")
    print("=" * 100)

    old_velocities = atoms.get_velocities()
    existing_positions = np.asarray(atoms.positions, dtype=float).copy()
    placed_positions = []
    used_min_dist_values = []
    placed = 0
    total_attempts = 0

    for mol_id in range(1, args.n_o2 + 1):
        current_min_dist = float(args.min_dist)
        placed_this = False

        for attempt in range(1, args.max_attempts_per_o2 + 1):
            total_attempts += 1
            if attempt > 1 and attempt % args.relax_after_attempts == 0 and current_min_dist > args.hard_min_dist:
                current_min_dist = max(args.hard_min_dist, current_min_dist - args.min_dist_step)
                print(f"[RELAX] O2 {mol_id}: insertion crowded; reducing min-dist to {current_min_dist:.3f} Å")

            fx, fy = rng.random(), rng.random()
            center_xy_cart = np.array([fx, fy, 0.0]) @ cell
            lz = local_surface_z(connected_positions, center_xy_cart, cell, args.local_radius)

            # Hybrid z rule:
            #   Stage A: preferentially place the O2 COM at local surface + preferred_height.
            #   Stage B: if crowded/unavailable after preferred attempts, fall back to random gas-region z.
            max_atom_z = wall_z - args.wall_clearance
            z_low = lz + args.surface_clearance + 0.5 * args.bond
            z_high = max_atom_z - 0.5 * args.bond
            if z_high <= z_low:
                continue

            if attempt <= args.preferred_attempts_per_o2:
                center_z = lz + args.preferred_height
                insertion_mode_this = "preferred_surface_height"
                # If preferred height would put any orientation too close to the wall region,
                # skip this trial and keep trying other x-y points. Fallback may later find
                # a lower random z if space exists.
                if center_z < z_low or center_z > z_high:
                    continue
            else:
                center_z = rng.uniform(z_low, z_high)
                insertion_mode_this = "random_below_wall_fallback"

            center_cart = np.array([fx, fy, 0.0]) @ cell
            center_cart[2] = center_z
            direction = random_unit_vector(rng)
            pair_positions = np.vstack([
                center_cart - 0.5 * args.bond * direction,
                center_cart + 0.5 * args.bond * direction,
            ])

            if np.min(pair_positions[:, 2]) < lz + args.surface_clearance:
                continue
            if np.max(pair_positions[:, 2]) > max_atom_z:
                continue

            all_existing = existing_positions
            if placed_positions:
                all_existing = np.vstack([existing_positions, np.asarray(placed_positions)])

            dmin = min_distance_to_existing(pair_positions, all_existing, cell)
            if dmin < current_min_dist:
                continue

            placed_positions.extend(pair_positions)
            used_min_dist_values.append(current_min_dist)
            placed += 1
            placed_this = True
            print(
                f"Inserted O2 {placed:3d}/{args.n_o2} | attempts_total={total_attempts:8d} | "
                f"attempts_this={attempt:7d} | mode={insertion_mode_this} | "
                f"local_surface_z={lz:8.3f} Å | center_z={center_z:8.3f} Å | "
                f"dmin={dmin:.3f} Å | min-dist-used={current_min_dist:.3f} Å"
            )
            break

        if not placed_this:
            raise RuntimeError(
                f"Failed to insert O2 {mol_id}/{args.n_o2} after {args.max_attempts_per_o2} attempts. "
                f"Minimum distance was relaxed down to {current_min_dist:.3f} Å. "
                "Try reducing --n-o2, increasing vacuum/wall height, or lowering --hard-min-dist cautiously."
            )

    new_symbols = atoms.get_chemical_symbols() + ["O"] * (2 * placed)
    new_positions = np.vstack([atoms.positions, np.asarray(placed_positions)])
    out_atoms = Atoms(symbols=new_symbols, positions=new_positions, cell=atoms.cell, pbc=atoms.pbc)
    out_atoms.info.update(atoms.info)
    out_atoms.info.update({
        "o2_added_this_stage": int(placed),
        "o_atoms_added_this_stage": int(2 * placed),
        "residual_free_o2_before_dosing": int(analysis["n_free_o2"]),
        "o2_insertion_seed": int(args.seed),
        "surface_detection": "connected_to_bottom_excluding_residual_free_o2",
        "insertion_mode": "hybrid_preferred_surface_height_then_random_below_wall",
        "preferred_height_A": float(args.preferred_height),
        "preferred_attempts_per_o2": int(args.preferred_attempts_per_o2),
        "surface_clearance_A": float(args.surface_clearance),
        "fermi_wall_z_A": float(wall_z),
        "wall_clearance_A": float(args.wall_clearance),
        "global_surface_z_A": float(global_surface_z),
        "min_dist_initial_A": float(args.min_dist),
        "hard_min_dist_A": float(args.hard_min_dist),
        "min_dist_used_min_A": float(np.min(used_min_dist_values)) if used_min_dist_values else None,
    })

    if not args.no_velocities:
        if old_velocities is not None:
            o2_atoms = Atoms(symbols=["O"] * (2 * placed), positions=np.asarray(placed_positions), cell=atoms.cell, pbc=atoms.pbc)
            MaxwellBoltzmannDistribution(o2_atoms, temperature_K=args.temperature, force_temp=True, rng=rng)
            new_velocities = np.vstack([old_velocities, o2_atoms.get_velocities()])
            if args.direct_o2_to_surface:
                new_velocities, flipped = direct_new_o2_com_velocity_toward_surface(new_velocities, len(atoms), placed)
                print(f"[INFO] Directed inserted O2 COM z-velocity downward for {flipped}/{placed} molecules by one-time mirroring.")
            out_atoms.set_velocities(new_velocities)
            print("[INFO] Preserved old velocities and initialized velocities for newly inserted O atoms only.")
        else:
            MaxwellBoltzmannDistribution(out_atoms, temperature_K=args.temperature, force_temp=True, rng=rng)
            if args.direct_o2_to_surface:
                new_velocities, flipped = direct_new_o2_com_velocity_toward_surface(out_atoms.get_velocities(), len(atoms), placed)
                out_atoms.set_velocities(new_velocities)
                print(f"[INFO] Directed inserted O2 COM z-velocity downward for {flipped}/{placed} molecules by one-time mirroring.")
            print("[INFO] Input had no velocities; initialized velocities for the whole output system.")
        if args.keep_stationary:
            Stationary(out_atoms)
            print("[INFO] Removed center-of-mass translation.")

    write_structure(args.output, out_atoms)

    print("=" * 100)
    print(f"Success: inserted {placed} O2 molecules; residual free O2 was kept, not purged.")
    print(f"Atom count: {len(atoms)} input -> {len(out_atoms)} output")
    print(f"Saved: {args.output}")
    print("=" * 100)


if __name__ == "__main__":
    main()
