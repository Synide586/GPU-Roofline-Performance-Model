import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = 4  # bytes per float


@dataclass
class Work:
    flops: float           # arithmetic actually executed
    dram: float = 0.0      # bytes to and from DRAM (compulsory)
    l2: float = 0.0        # bytes served by L2 beyond the compulsory traffic
    l1: float = 0.0        # L1/shared traffic, counted as 128 bytes per warp access to one cache line


def load_ceilings(path):
    c = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            c[row["name"]] = float(row["measured"]) * 1e9  # GFLOP/s or GB/s -> per second
    return {
        "compute": c["fp32_fma"],
        "dram": c["dram_triad"],  # kernels here both read and write: the mixed-traffic ceiling applies
        "l2": c["l2_read"],
        "l1": c["smem_read"],     # L1 and shared memory are the same on-chip unit
    }


def predict(work, ceil):
    times = {
        "compute": work.flops / ceil["compute"],
        "DRAM": work.dram / ceil["dram"],
        "L2": work.l2 / ceil["l2"],
        "L1/shared": work.l1 / ceil["l1"],
    }
    bound = max(times, key=times.get)
    return times[bound], bound


# ------------------------------- GEMM --------------------------------------
# bytes of L1/shared traffic per FMA, from each kernel's access pattern (no bank conflicts assumed)
GEMM_L1_PER_FMA = {
    "naive_uncoalesced": 33 * 128 / 32,  # per warp step: A hits 32 different lines, B is one broadcast
    "naive_coalesced": 2 * 128 / 32,     # per warp step: A broadcast + B one contiguous line
    "smem_tiled": 2 * 128 / 32,          # two shared loads per FMA
    "2d_blocktiling": 16 * 128 / 2048,   # 16 shared loads per warp feed 32 x 64 FMAs
    "2d_blocktiling_occ": 16 * 128 / 2048,
    "2d_blocktiling_v2": 4 * 512 / 2048,  # four conflict-free 128-bit loads per warp feed 32 x 64 FMAs
    "cublas": 0.0,                       # internals unknown: compute and DRAM only
}


def gemm_work(kernel, n):
    fma = float(n) ** 3
    return Work(flops=2 * fma, dram=3 * n * n * F, l1=GEMM_L1_PER_FMA[kernel] * fma)


# ------------------------- softmax / LayerNorm -----------------------------
ROWNORM_FLOPS_PER_ELEM = {"softmax": 5, "layernorm": 8}  # approximate; only used to place points on the plot
ROWNORM_MODELED = {"softmax_block", "softmax_warp_online", "softmax_warp_reg",
                   "layernorm_block", "layernorm_warp_reg", "torch_softmax", "torch_layernorm"}


def rownorm_work(op, rows, cols):
    n = rows * cols
    return Work(flops=ROWNORM_FLOPS_PER_ELEM[op] * n, dram=2 * n * F)  # read x once, write y once


# ------------------------------ attention ----------------------------------
D, TILE = 64, 64


def attention_pairs(n, causal, kernel):
    """(query, key) pairs actually computed per head."""
    if not causal:
        return n * n
    if kernel == "flash_v2":  # whole 64 x 64 tiles on and below the diagonal
        t = n // TILE
        return TILE * TILE * t * (t + 1) / 2
    return n * (n + 1) / 2


def attention_work(kernel, bh, n, causal):
    qkvo = 4 * bh * n * D * F
    if kernel == "unfused_cublas":  # three kernels run one after another: return one Work per stage
        s = bh * n * n * F
        gemm = 2 * bh * n * n * D
        return [Work(flops=gemm, dram=2 * bh * n * D * F + s),   # S = Q K^T, written to DRAM
                Work(flops=5 * bh * n * n, dram=2 * s),          # softmax reads and rewrites S
                Work(flops=gemm, dram=s + 2 * bh * n * D * F)]   # O = P V reads S again
    pairs = bh * attention_pairs(n, causal, kernel)
    fma = pairs * 2 * D  # D FMAs for the score, D for the output
    if kernel == "flash_v1":
        l1 = fma * 128 / 32  # one broadcast shared load per FMA
        kv_reads = 0.0
    else:
        l1 = fma * 8 * 128 / 512  # 8 shared loads per warp feed 16 x 32 FMAs
        tiles = n // TILE if not causal else (n // TILE + 1) / 2
        kv_reads = bh * tiles * 2 * n * D * F  # K and V re-read once per query tile, mostly from L2
    return [Work(flops=2 * fma, dram=qkvo, l2=kv_reads, l1=l1)]


def attention_useful_flops(bh, n, causal):
    return 4 * bh * n * n * D * (0.5 if causal else 1.0)


# --------------------------------- main ------------------------------------
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceilings", default="results/ceilings.csv")
    ap.add_argument("--data", default="data/project1")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    ceil = load_ceilings(args.ceilings)
    print("Ceilings: compute %.0f GFLOP/s | DRAM %.0f GB/s | L2 %.0f GB/s | L1/shared %.0f GB/s\n"
          % (ceil["compute"] / 1e9, ceil["dram"] / 1e9, ceil["l2"] / 1e9, ceil["l1"] / 1e9))

    results = []  # (group, kernel, shape, predicted_ms, measured_ms, bound, useful_flops, dram_bytes, l1_bytes)

    for r in read_csv(os.path.join(args.data, "gemm.csv")):
        n, k = int(r["size"]), r["kernel"]
        w = gemm_work(k, n)
        t, bound = predict(w, ceil)
        results.append(("gemm", k, f"N={n}", t * 1e3, float(r["ms"]), bound, w.flops, w.dram, w.l1))

    for r in read_csv(os.path.join(args.data, "rownorm.csv")):
        if r["kernel"] not in ROWNORM_MODELED:
            continue
        rows, cols = int(r["rows"]), int(r["cols"])
        w = rownorm_work(r["op"], rows, cols)
        t, bound = predict(w, ceil)
        results.append((r["op"], r["kernel"], f"{rows}x{cols}", t * 1e3, float(r["ms"]), bound, w.flops, w.dram, 0.0))

    for r in read_csv(os.path.join(args.data, "attention.csv")):
        k, n, bh, causal = r["kernel"], int(r["n"]), int(r["bh"]), r["causal"] == "1"
        if k == "torch_sdpa":
            continue
        stages = attention_work(k, bh, n, causal)
        preds = [predict(w, ceil) for w in stages]
        t = sum(p[0] for p in preds)
        bound = "+".join(p[1] for p in preds) if len(preds) > 1 else preds[0][1]
        dram = sum(w.dram for w in stages)
        l1 = sum(w.l1 for w in stages)
        results.append(("attention", k, f"N={n}{' causal' if causal else ''}", t * 1e3, float(r["ms"]),
                        bound, attention_useful_flops(bh, n, causal), dram, l1))

    print(f"{'kernel':<20} {'shape':<16} {'bound':<26} {'model ms':>10} {'measured':>10} {'% of bound':>11}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model_vs_measured.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "kernel", "shape", "bound", "predicted_ms", "measured_ms", "pct_of_bound"])
        for g, k, shape, pred, meas, bound, *_ in results:
            pct = 100 * pred / meas
            w.writerow([g, k, shape, bound, f"{pred:.4f}", f"{meas:.4f}", f"{pct:.1f}"])
            if g != "gemm" or shape == "N=4096":
                if g in ("softmax", "layernorm") and not shape.endswith("x1024"):
                    continue
                if g == "attention" and not shape.startswith("N=4096"):
                    continue
                print(f"{k:<20} {shape:<16} {bound:<26} {pred:>10.3f} {meas:>10.3f} {pct:>10.0f}%")
    print(f"\nFull table: {os.path.join(args.out, 'model_vs_measured.csv')}")

    plot(results, ceil, os.path.join(args.out, "roofline.png"))


PLOT_ROWNORM = {"softmax_block", "softmax_warp_online", "softmax_warp_reg", "layernorm_warp_reg",
                "torch_softmax", "torch_layernorm"}


def plot(results, ceil, path):
    picks = [r for r in results
             if (r[0] == "gemm" and r[2] == "N=4096")
             or (r[0] in ("softmax", "layernorm") and r[2].endswith("x1024") and r[1] in PLOT_ROWNORM)
             or (r[0] == "attention" and r[2] == "N=4096")]
    names = sorted({r[1] for r in picks})
    colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(names)}
    markers = {"gemm": "o", "softmax": "s", "layernorm": "D", "attention": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    xs = [10 ** (i / 20) for i in range(-50, 81)]
    peak = ceil["compute"] / 1e9
    for ax, level, label in [(axes[0], "dram", "DRAM"), (axes[1], "l1", "L1/shared")]:
        bw = ceil[level] / 1e9
        ax.plot(xs, [min(peak, x * bw) for x in xs], "k-", lw=2, label=f"{label} roof {bw:.0f} GB/s")
        if level == "dram":
            l2 = ceil["l2"] / 1e9
            ax.plot(xs, [min(peak, x * l2) for x in xs], "k:", lw=1, label=f"L2 roof {l2:.0f} GB/s")
        for g, k, shape, pred, meas, bound, flops, dram, l1 in picks:
            nbytes = dram if level == "dram" else l1
            if nbytes > 0:
                ax.scatter(flops / nbytes, flops / (meas * 1e6), s=55, color=colors[k], marker=markers[g],
                           edgecolors="black", linewidths=0.5, label=k)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Arithmetic intensity (FLOPs per {label} byte)")
        ax.set_ylabel("Achieved GFLOP/s")
        ax.set_title(f"{label} roofline, compute roof {peak:.0f} GFLOP/s")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left" if level == "dram" else "lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
