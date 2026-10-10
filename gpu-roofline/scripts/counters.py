import argparse
import csv
import io
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: E402

METRICS = [
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "l1tex__data_pipe_lsu_wavefronts.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
]
AVERAGED = {"sm__warps_active.avg.pct_of_peak_sustained_active"}

GEMM_N, ROWS, COLS, ATTN_N, BH = 2048, 8192, 1024, 2048, 16

CASES = [
    ("gemm", "naive_uncoalesced", ["bench_gemm", "--profile", "naive_uncoalesced", "--sizes", str(GEMM_N)]),
    ("gemm", "naive_coalesced", ["bench_gemm", "--profile", "naive_coalesced", "--sizes", str(GEMM_N)]),
    ("gemm", "smem_tiled", ["bench_gemm", "--profile", "smem_tiled", "--sizes", str(GEMM_N)]),
    ("gemm", "2d_blocktiling", ["bench_gemm", "--profile", "2d_blocktiling", "--sizes", str(GEMM_N)]),
    ("gemm", "2d_blocktiling_occ", ["bench_gemm", "--profile", "2d_blocktiling_occ", "--sizes", str(GEMM_N)]),
    ("gemm", "2d_blocktiling_v2", ["bench_gemm", "--profile", "2d_blocktiling_v2", "--sizes", str(GEMM_N)]),
    ("gemm", "cublas", ["bench_gemm", "--profile", "cublas", "--sizes", str(GEMM_N)]),
    ("softmax", "softmax_block", ["bench_rownorm", "--profile", "softmax_block", "--cols", str(COLS)]),
    ("softmax", "softmax_warp_online", ["bench_rownorm", "--profile", "softmax_warp_online", "--cols", str(COLS)]),
    ("softmax", "softmax_warp_reg", ["bench_rownorm", "--profile", "softmax_warp_reg", "--cols", str(COLS)]),
    ("layernorm", "layernorm_warp_reg", ["bench_rownorm", "--profile", "layernorm_warp_reg", "--cols", str(COLS)]),
    ("attention", "unfused_cublas", ["bench_attention", "--profile", "unfused_cublas", "--seqs", str(ATTN_N)]),
    ("attention", "flash_v1", ["bench_attention", "--profile", "flash_v1", "--seqs", str(ATTN_N)]),
    ("attention", "flash_v2", ["bench_attention", "--profile", "flash_v2", "--seqs", str(ATTN_N)]),
]


def run_ncu(bin_dir, cmd, metrics):
    """Profile every kernel the command launches; drop metrics this GPU or ncu version does not have."""
    metrics = list(metrics)
    while True:
        full = ["ncu", "--csv", "--print-units", "base", "--metrics", ",".join(metrics),
                os.path.join(bin_dir, cmd[0])] + cmd[1:]
        p = subprocess.run(full, capture_output=True, text=True)
        out = p.stdout + p.stderr
        missing = re.findall(r"[Ff]ailed to find metric '?([\w.]+)'?", out)
        if missing and any(m in metrics for m in missing):
            for m in missing:
                if m in metrics:
                    print(f"    (metric not available here, skipped: {m})")
                    metrics.remove(m)
            continue
        if p.returncode != 0:
            raise RuntimeError(f"ncu failed for {' '.join(cmd)}:\n{out[-2000:]}")
        return parse(p.stdout)


def parse(text):
    lines = [l for l in text.splitlines() if l.startswith('"')]
    totals, counts = {}, {}
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        name = row.get("Metric Name")
        raw = (row.get("Metric Value") or "").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + v  # several kernels (cuBLAS, unfused attention): add them up
        counts[name] = counts.get(name, 0) + 1
    for name in AVERAGED & totals.keys():
        totals[name] /= counts[name]
    return totals


def model_work(group, kernel):
    if group == "gemm":
        return model.gemm_work(kernel, GEMM_N)
    if group in ("softmax", "layernorm"):
        return model.rownorm_work(group, ROWS, COLS)
    stages = model.attention_work(kernel, BH, ATTN_N, False)
    return model.Work(flops=sum(w.flops for w in stages), dram=sum(w.dram for w in stages),
                      l2=sum(w.l2 for w in stages), l1=sum(w.l1 for w in stages))


def fmt(x, unit=""):
    if x is None:
        return "-"
    for div, suffix in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if abs(x) >= div:
            return f"{x / div:.2f}{suffix}{unit}"
    return f"{x:.0f}{unit}"


def ratio(measured, modeled):
    return f"{measured / modeled:.2f}x" if measured is not None and modeled else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels-repo", default="../cuda-transformer-kernels")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="", help="comma-separated kernel names to profile")
    args = ap.parse_args()
    bin_dir = os.path.join(args.kernels_repo, "bin")
    if not os.path.isdir(bin_dir):
        sys.exit(f"{bin_dir} not found: build the kernels first (cd {args.kernels_repo} && make)")
    only = set(filter(None, args.only.split(",")))

    rows = []
    for group, kernel, cmd in CASES:
        if only and kernel not in only:
            continue
        print(f"profiling {kernel} ...", flush=True)
        m = run_ncu(bin_dir, cmd, METRICS)
        g = m.get
        flops = None
        if g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") is not None:
            flops = (2 * g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
                     + g("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
                     + g("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0))
        dram = None
        if g("dram__bytes_read.sum") is not None:
            dram = g("dram__bytes_read.sum", 0) + g("dram__bytes_write.sum", 0)
        smem_wf = g("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
        wf = g("l1tex__data_pipe_lsu_wavefronts.sum", smem_wf)
        l1 = wf * 128 if wf is not None else None
        conflicts = g("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
        w = model_work(group, kernel)
        rows.append({
            "kernel": kernel, "group": group,
            "flops_model": w.flops, "flops_measured": flops,
            "dram_model": w.dram, "dram_measured": dram,
            "l2_measured": g("lts__t_bytes.sum"),
            "l1_model": w.l1, "l1_measured": l1,
            "smem_wavefronts": smem_wf, "smem_bank_conflicts": conflicts,
            "occupancy_pct": g("sm__warps_active.avg.pct_of_peak_sustained_active"),
            "duration_ms_base_clock": (g("gpu__time_duration.sum") or 0) / 1e6 or None,
        })

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "counters.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\nGEMM N={GEMM_N}, softmax/LayerNorm {ROWS}x{COLS}, attention BH={BH} N={ATTN_N} (non-causal)")
    print("measured / model ratios: 1.00x means the model counted exactly what the hardware did\n")
    print(f"{'kernel':<20} {'FLOPs':>9} {'meas/model':>10} {'DRAM':>9} {'meas/model':>10} "
          f"{'L1/shared':>10} {'meas/model':>10} {'bank confl.':>11} {'occupancy':>9}")
    for r in rows:
        occ = f"{r['occupancy_pct']:.0f}%" if r["occupancy_pct"] is not None else "-"
        print(f"{r['kernel']:<20} {fmt(r['flops_measured']):>9} {ratio(r['flops_measured'], r['flops_model']):>10} "
              f"{fmt(r['dram_measured'], 'B'):>9} {ratio(r['dram_measured'], r['dram_model']):>10} "
              f"{fmt(r['l1_measured'], 'B'):>10} {ratio(r['l1_measured'], r['l1_model']):>10} "
              f"{fmt(r['smem_bank_conflicts']):>11} {occ:>9}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
