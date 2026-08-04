#!/usr/bin/env python3
import argparse
import sys
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CUTOFFS = {
    ("O", "Si"): 1.80,      # 已排序，确保 get_cutoff 能匹配
    ("Fe", "O"): 2.20,
    ("Fe", "Si"): 2.80,
}

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute the largest bridged cluster from a VASP XDATCAR trajectory. "
            "Example: Si-O-Si or Fe-O-Fe. Nodes are species A; two A atoms are "
            "connected if they share at least one bridge species B within cutoff."
        )
    )
    p.add_argument("node_species", help="Node species A, e.g. Si or Fe")
    p.add_argument("bridge_species", help="Bridge species B, e.g. O")
    p.add_argument("nsteps", type=int, help="Total number of frames to read")
    p.add_argument("discard", type=int, help="Number of initial frames to discard")
    p.add_argument("--xdatcar", default="XDATCAR", help="Path to XDATCAR")
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="A-B bonding cutoff in angstrom. If omitted, a default is used when available.",
    )
    p.add_argument("--dt", type=float, default=1.0, help="Time step per frame in fs (仅用于参考，不输出)")
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix. Default: bridge_cluster_<A>_<B>_<A>",
    )
    p.add_argument(
        "--min-angle",
        type=float,
        default=None,
        help=(
            "Optional A-B-A angle threshold in degrees. If provided, two A atoms are "
            "connected through B only when angle(A-B-A) >= min-angle."
        ),
    )
    return p.parse_args()

class XdatcarReader:
    def __init__(self, path):
        self.path = Path(path)
        self._parse_header()

    def _parse_header(self):
        with self.path.open("r") as f:
            self.comment = f.readline().rstrip("\n")
            if not self.comment:
                raise ValueError("Invalid XDATCAR: empty first line")
            self.scale = float(f.readline().split()[0])
            lattice = []
            for _ in range(3):
                lattice.append([float(x) for x in f.readline().split()[:3]])
            self.lattice = np.array(lattice, dtype=float) * self.scale
            self.species = f.readline().split()
            self.counts = [int(x) for x in f.readline().split()]
            self.natoms = sum(self.counts)
            self.indices_by_species = {}
            start = 0
            for sp, count in zip(self.species, self.counts):
                idx = np.arange(start, start + count, dtype=int)
                self.indices_by_species[sp] = idx
                start += count
            self.header_lines = 7

    def iter_frames(self, nsteps=None):
        with self.path.open("r") as f:
            for _ in range(self.header_lines):
                next(f)
            step_counter = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if not line.lower().startswith(("direct configuration", "cartesian configuration")):
                    raise ValueError(f"Unexpected frame header: {line.strip()}")
                coords = np.empty((self.natoms, 3), dtype=float)
                for i in range(self.natoms):
                    parts = f.readline().split()
                    if len(parts) < 3:
                        raise ValueError(
                            f"Unexpected end of coordinate block in frame {step_counter + 1}"
                        )
                    coords[i] = [float(x) for x in parts[:3]]  # 只取前三个坐标
                step_counter += 1
                yield step_counter, coords
                if nsteps is not None and step_counter >= nsteps:
                    break

def get_cutoff(a, b, user_cutoff=None):
    if user_cutoff is not None:
        return float(user_cutoff)
    key = tuple(sorted((a, b)))
    if key in DEFAULT_CUTOFFS:
        return DEFAULT_CUTOFFS[key]
    raise ValueError(f"No default cutoff for pair {a}-{b}. Please provide --cutoff explicitly.")

def minimum_image_deltas(frac_a, frac_b):
    d = frac_a[:, None, :] - frac_b[None, :, :]
    d -= np.round(d)
    return d

def pair_distance_matrix(frac_a, frac_b, lattice):
    d_frac = minimum_image_deltas(frac_a, frac_b)
    d_cart = np.einsum("...j,jk->...k", d_frac, lattice)
    return np.linalg.norm(d_cart, axis=-1)

def vectors_from_bridge_to_nodes(frac_nodes, frac_bridge, lattice):
    d_frac = frac_nodes[:, None, :] - frac_bridge[None, :, :]
    d_frac -= np.round(d_frac)
    return np.einsum("...j,jk->...k", d_frac, lattice)

def angle_deg(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = np.dot(v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def largest_bridged_cluster(frame_frac, lattice, idx_node, idx_bridge, cutoff, min_angle=None):
    frac_node = frame_frac[idx_node]
    frac_bridge = frame_frac[idx_bridge]
    n_node = len(idx_node)
    n_bridge = len(idx_bridge)
    if n_node == 0 or n_bridge == 0:
        return 0, 0, 0.0

    dist = pair_distance_matrix(frac_node, frac_bridge, lattice)
    node_neighbors_of_bridge = [np.where(dist[:, j] < cutoff)[0].tolist() for j in range(n_bridge)]

    if min_angle is not None:
        vecs = vectors_from_bridge_to_nodes(frac_node, frac_bridge, lattice)
    else:
        vecs = None

    adjacency = [set() for _ in range(n_node)]
    bridge_to_valid_pairs = []

    for j, node_list in enumerate(node_neighbors_of_bridge):
        if len(node_list) < 2:
            bridge_to_valid_pairs.append([])
            continue
        valid_pairs = []
        m = len(node_list)
        for a_idx in range(m):
            ia = node_list[a_idx]
            for b_idx in range(a_idx + 1, m):
                ib = node_list[b_idx]
                if min_angle is not None:
                    ang = angle_deg(vecs[ia, j], vecs[ib, j])
                    if ang < min_angle:
                        continue
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)
                valid_pairs.append((ia, ib))
        bridge_to_valid_pairs.append(valid_pairs)

    visited = np.zeros(n_node, dtype=bool)
    best_nodes = []
    for s in range(n_node):
        if visited[s]:
            continue
        q = deque([s])
        visited[s] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        if len(comp) > len(best_nodes):
            best_nodes = comp

    best_node_set = set(best_nodes)
    bridge_count = 0
    for j, node_list in enumerate(node_neighbors_of_bridge):
        if len(node_list) < 2:
            continue
        if min_angle is None:
            count_in = sum(1 for i in node_list if i in best_node_set)
            if count_in >= 2:
                bridge_count += 1
        else:
            if any((ia in best_node_set and ib in best_node_set) for ia, ib in bridge_to_valid_pairs[j]):
                bridge_count += 1

    cluster_size_nodes = len(best_nodes)
    cluster_fraction = cluster_size_nodes / n_node if n_node > 0 else 0.0
    return cluster_size_nodes, bridge_count, cluster_fraction

def main():
    args = parse_args()
    a = args.node_species
    b = args.bridge_species
    cutoff = get_cutoff(a, b, args.cutoff)
    reader = XdatcarReader(args.xdatcar)

    if a not in reader.indices_by_species:
        raise ValueError(f"Species {a} not found in XDATCAR. Available: {reader.species}")
    if b not in reader.indices_by_species:
        raise ValueError(f"Species {b} not found in XDATCAR. Available: {reader.species}")

    idx_node = reader.indices_by_species[a]
    idx_bridge = reader.indices_by_species[b]

    out_prefix = args.out_prefix or f"bridge_cluster_{a}_{b}_{a}"
    csv_path = Path(f"{out_prefix}.csv")
    png_path = Path(f"{out_prefix}.png")

    results = []
    read_frames = 0

    for step, frame_frac in reader.iter_frames(nsteps=args.nsteps):
        read_frames += 1
        if step <= args.discard:
            continue
        n_nodes, n_bridges, frac = largest_bridged_cluster(
            frame_frac,
            reader.lattice,
            idx_node,
            idx_bridge,
            cutoff,
            min_angle=args.min_angle,
        )
        results.append((step, n_nodes, n_bridges, frac))

    if read_frames < args.nsteps:
        print(
            f"Warning: requested {args.nsteps} frames, but XDATCAR contains only {read_frames} frames.",
            file=sys.stderr,
        )

    # 写入 CSV（无 time_fs）
    with csv_path.open("w") as f:
        f.write(
            f"step,max_cluster_{a}_count,max_cluster_{b}_bridge_count,max_cluster_fraction_{a}\n"
        )
        for step, n_nodes, n_bridges, frac in results:
            f.write(f"{step},{n_nodes},{n_bridges},{frac:.8f}\n")

    # 画图并保存（单 Y 轴）
    if results:
        df = pd.DataFrame(
            results,
            columns=["step", f"max_cluster_{a}_count", f"max_cluster_{b}_bridge_count", f"max_cluster_fraction_{a}"]
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            df["step"], 
            df[f"max_cluster_{a}_count"], 
            color="blue", 
            linewidth=1.8, 
            label=f"{a} atoms in largest cluster"
        )
        plt.plot(
            df["step"], 
            df[f"max_cluster_{b}_bridge_count"], 
            color="darkorange", 
            linewidth=1.8, 
            linestyle="--", 
            label=f"{b} bridges in largest cluster"
        )

        plt.xlabel("Step")
        plt.ylabel("Number of atoms / bridges in largest cluster")
        plt.title(f"Evolution of largest {a}-{b}-{a} cluster\n(cutoff = {cutoff:.2f} Å)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()

        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.show()

    # 打印总结
    print(f"\nBridge network : {a}-{b}-{a}")
    print(f"A-B cutoff (Å) : {cutoff:.3f}")
    print(f"Min angle (deg): {args.min_angle if args.min_angle is not None else 'None'}")
    print(f"Frames requested: {args.nsteps}")
    print(f"Discarded frames: {args.discard}")
    print(f"Frames analyzed : {len(results)}")
    print(f"Output CSV      : {csv_path}")
    print(f"Output figure   : {png_path}")
    if results:
        max_nodes = max(x[1] for x in results)
        print(f"Maximum {a} cluster size: {max_nodes}")

if __name__ == "__main__":
    main()
