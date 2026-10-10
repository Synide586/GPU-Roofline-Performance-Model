import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/bandwidth_sweep.csv"
    sizes, gbs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            sizes.append(int(row["bytes"]) / 2**20)
            gbs.append(float(row["gbs"]))

    plt.figure(figsize=(8, 4.5))
    plt.plot(sizes, gbs, marker="o")
    plt.axvline(4, color="gray", linestyle="--", label="L2 size (4 MB on T4)")
    plt.xscale("log", base=2)
    plt.xlabel("Working-set size (MB)")
    plt.ylabel("Read bandwidth (GB/s)")
    plt.title("Read bandwidth vs working-set size: L2 vs DRAM")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
