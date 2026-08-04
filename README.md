# beta-FeSi2-oxidation-MACE

Scripts and input data for MACE molecular-dynamics simulations of sequential O2 dosing and oxidation on beta-FeSi2 surfaces.

## Repository contents

### Sequential O2 dosing workflow

- `sequential_O2_driver_hybrid_final15.py` — main residual-O2-aware pulse-dosing driver.
- `add_O2_hybrid_surface_then_wall.py` — inserts O2 using a preferred local-surface height with a below-wall fallback.
- `run_MD_constrain_continue_batch.py` — runs batch/continuation MACE committee MD with a fixed bottom region and an upper Fermi-like wall.

### Additional MD and analysis scripts

- `run_MD_constrain.py` — standalone/legacy constrained MACE MD script.
- `count_o2.py` — analyzes the remaining free O2 fraction in two trajectories.
- `max_cluster_bridge.py` — computes the largest bridged cluster, such as Si-O-Si or Fe-O-Fe.
- `plot_maxcluster_new.py` — combines cluster CSV files, reports late-stage statistics, and plots normalized cluster evolution.
- `FeSi2.extxyz` — beta-FeSi2 input structure/data file.

## Main workflow

For each new O2 molecule, the insertion script first attempts

```text
center_z = local_connected_surface_z + preferred_height
```

where the default `preferred_height` is 2.50 Å. If this position is crowded, overlaps residual O2, or violates the upper-wall safety margin, the script falls back to random insertion in the available gas region:

```text
lower_z = local_connected_surface_z + surface_clearance
upper_z = Fermi_wall_z - wall_clearance
```

All existing atoms, including residual gas-phase O2, are included in overlap checks. When insertion is difficult, the minimum-distance threshold can be reduced gradually from `--min-dist` to `--hard-min-dist`, but never below the hard limit.

After every pulse and main MD stage, residual free O2 is counted:

- If residual O2 is at or below `--residual-threshold`, the next pulse is inserted.
- Otherwise, extension MD chunks are run while the residual count decreases by at least `--min-residual-decrease`, subject to `--max-extensions`.
- After the target exposure is reached, a final relaxation is run for `--final-md-steps`.

At a 1 fs timestep, `--final-md-steps 15000` corresponds to 15 ps.

## Requirements

A typical Python environment needs:

```bash
pip install numpy scipy pandas matplotlib tqdm ase mace-torch
```

MACE model weight files are not included in this repository. Put them in `MACE_models/` or supply another directory with `--model-dir`.

The default model filenames expected by the MD scripts are:

```text
FeSiO_com1_run-123_stagetwo.model
FeSiO_com2_run-456_stagetwo.model
```

## Example command

Run from the repository root:

```bash
python sequential_O2_driver_hybrid_final15.py \
  --initial FeSi2.extxyz \
  --model-dir /path/to/MACE_models \
  --total-o2 104 \
  --o2-per-round 8 \
  --md-steps 3000 \
  --extension-steps 1000 \
  --final-md-steps 15000 \
  --residual-threshold 1 \
  --min-residual-decrease 1 \
  --preferred-height 2.5 \
  --preferred-attempts-per-o2 50000 \
  --surface-clearance 1.50 \
  --local-radius 4.0 \
  --min-dist 1.78 \
  --hard-min-dist 1.55 \
  --min-dist-step 0.05 \
  --temperature 1200 \
  --upper-zfrac 0.65 \
  --wall-clearance 2.0 \
  --wall-upper-center-frac 0.65 \
  --direct-o2-to-surface \
  --workdir sequential_104O2_hybrid_final15_Si_1200K
```

## Notes

- `--preferred-height` controls the first-choice O2 center height above the local connected surface.
- `--surface-clearance` is the lower safety bound used during fallback insertion and overlap checks.
- `--wall-clearance` keeps inserted O2 below the Fermi-like upper wall.
- `--direct-o2-to-surface` mirrors the COM z-velocity of newly inserted O2 downward once at insertion; it does not add a continuing acceleration or force during MD.
- The standalone `run_MD_constrain.py` accepts `MACE_MODEL_DIR` and `FESI2_INIT_XYZ` environment variables for portable paths.

## License

MIT License. See `LICENSE`.
