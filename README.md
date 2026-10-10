# GPU Roofline Performance Model

How fast *should* a GPU kernel be, and why isn't it that fast? This project measures the real limits of a GPU with microbenchmarks, builds a roofline model from them, and uses it to explain the performance of my hand-written CUDA kernels from [cuda-transformer-kernels](https://github.com/Synide586/CUDA-Kernals-for-Transformer-ops).

## The roofline idea

Every kernel does some arithmetic (FLOPs) and moves some data (bytes). The ratio is its **arithmetic intensity** (FLOPs per byte). The best possible speed is

`attainable FLOP/s = min(peak compute, arithmetic intensity x memory bandwidth)`

Kernels with low intensity are limited by memory bandwidth; kernels with high intensity are limited by compute. The intensity where the two limits meet is the **ridge point**.

## Part 1: measuring the ceilings

Datasheet peaks assume ideal conditions (for example the maximum boost clock), so the ceilings are measured with microbenchmarks:

| Benchmark | What it measures | How |
|---|---|---|
| FP32 FMA | peak compute | 8 independent FMA chains per thread, so the pipeline never stalls |
| DRAM read / write / copy / triad | main-memory bandwidth | 256 MB buffers (much larger than L2), 128-bit accesses |
| Working-set sweep | L2 vs DRAM bandwidth | repeated reads of 0.5 MB to 256 MB buffers, bypassing L1 |
| Shared-memory read | on-chip bandwidth | conflict-free 128-bit shared-memory loads |

Results on a Tesla T4:

| Ceiling | Measured | Datasheet | Note |
|---|---|---|---|
| FP32 compute | 7,504 GFLOP/s | 8,141 | the T4 sustains about 1.47 GHz, not its 1.59 GHz boost clock |
| DRAM read | 279 GB/s | 320 | |
| DRAM copy / triad (read + write) | 229 / 240 GB/s | 320 | mixing reads and writes costs about 15% |
| L2 read | 1,193 GB/s | | about 4.3x DRAM (best rate for buffers up to half the L2 size) |
| Shared memory | 4,000 GB/s | 4,070 | 64 bytes per clock per SM on Turing |

The ridge point is about **27 FLOPs per DRAM byte**: below it a kernel is limited by memory, above it by compute.


The sweep shows the cache hierarchy directly: about 1.2 TB/s while the data fits in the 4 MB L2, then a drop to about 270 GB/s once it has to come from DRAM.

Two lessons from building these benchmarks:

- **A result above the hardware limit means the benchmark is wrong.** The first shared-memory test reported 173% of peak: the compiler had merged repeated loads, so half the counted bytes were never loaded. Inspecting the compiled code (SASS) found it.
- **A benchmark must use the data the way you think it does.** The first L2 sweep showed the cliff at 16-32 MB instead of 4 MB, because blocks that were not running at the same time never shared a working set. Launching only resident blocks fixed it.

## Part 2: the model

For each kernel from [cuda-transformer-kernels](https://github.com/Synide586/CUDA-Kernals-for-Transformer-ops), `scripts/model.py` counts the FLOPs and the bytes moved at each level of the memory hierarchy (DRAM, L2, L1/shared), predicts the time each level needs from the measured ceilings, and takes the slowest one as the bound. Comparing that bound with the measured time shows how close each kernel gets and what limits it.

Shared-memory and L1 traffic is counted as 128 bytes per warp access to one cache line, assuming no bank conflicts. The measured times are in `data/project1/`.

Results at N = 4096 (GEMM and attention) and 8192 x 1024 (softmax, LayerNorm):

| Kernel | Bound | Model (ms) | Measured (ms) | % of bound |
|---|---|---|---|---|
| naive_uncoalesced GEMM | L1 | 2,267.6 | 2,215.2 | 102% |
| naive_coalesced GEMM | L1 | 137.4 | 232.2 | 59% |
| smem_tiled GEMM | shared memory | 137.4 | 152.8 | 90% |
| 2d_blocktiling GEMM | compute | 18.3 | 40.7 | 45% |
| cuBLAS GEMM | compute | 18.3 | 32.0 | 57% |
| softmax_warp_reg | DRAM | 0.280 | 0.301 | 93% |
| layernorm_block | DRAM | 0.280 | 0.287 | 98% |
| softmax_warp_online | DRAM | 0.280 | 0.413 | 68% |
| unfused attention | DRAM (3 kernels) | 18.2 | 32.7 | 56% |
| flash_v1 | shared memory | 34.4 | 38.1 | 90% |
| flash_v2 | shared memory | 17.2 | 24.9 | 69% |

What it shows:

- **The coalescing gap is explained by cache lines alone.** The uncoalesced GEMM touches 33 cache lines per step where the coalesced one touches 2; that count predicts its time to within 2%.
- **The simple kernels are at their limit.** Tiled GEMM, flash_v1 and the bandwidth-bound softmax/LayerNorm kernels reach 90-98% of their bound; to go faster they must move less data, not run more efficiently.
- **The fastest kernels are not at a bandwidth or compute limit.** The register-blocked GEMM (45%) and flash_v2 (69%) leave a gap the model cannot explain with traffic alone, pointing to latency and occupancy instead (Part 3 checks this with hardware counters).
- **Fewer passes did not mean faster.** Online softmax reads memory less but reaches only 68% of the DRAM bound, because its serial dependency chain keeps too few loads in flight.

## Part 3: checking the model with hardware counters

A model is only as good as its assumptions. `scripts/counters.py` runs each kernel under Nsight Compute and reads the hardware counters for the quantities the model estimates: FP32 instructions (FLOPs), DRAM bytes, L1/shared-memory wavefronts (one wavefront = one 128-byte access), shared-memory bank conflicts and occupancy. It then prints measured / modeled ratios: 1.00x means the model counted exactly what the hardware did.

Results (GEMM N = 2048, softmax/LayerNorm 8192 x 1024, attention 16 heads x N = 2048):

| Kernel | FLOPs | DRAM | L1/shared | Bank conflicts | Occupancy |
|---|---|---|---|---|---|
| naive_uncoalesced | 1.00x | 44.9x | 1.00x | 0 | 98% |
| naive_coalesced | 1.00x | 33.6x | 1.00x | 0 | 99% |
| smem_tiled | 1.00x | 33.2x | 0.82x | 0 | 100% |
| 2d_blocktiling | 1.00x | 4.5x | 1.49x | 33.6M | 25% |
| cuBLAS | 1.00x | 6.5x | - | 0.76M | 48% |
| softmax_warp_reg | 2.66x | 1.17x | - | 0 | 90% |
| layernorm_warp_reg | 1.00x | 1.14x | - | 0 | 67% |
| unfused attention | 1.08x | 1.40x | - | 3.4M | 61% |
| flash_v1 | 1.05x | 2.12x | 0.55x | 0.07M | 12% |
| flash_v2 | 1.07x | 1.30x | 0.94x | 3.3M | 25% |

Measured / modeled; 1.00x means the model counted exactly what the hardware did.

What the counters show:

- **FLOPs and cache-line counting are exact for the simple kernels.** The GEMM FLOP counts and the L1 traffic of both naive kernels match the model to 1.00x, which confirms that the 54x coalescing gap comes from cache lines touched.
- **The compiler changes the traffic.** The SASS shows that shared-memory reads of 4 consecutive values were merged into single 128-bit loads (`LDS.U.128`), so smem_tiled (0.82x) and flash_v1 (0.55x) move less shared-memory traffic than the source suggests. A model built from source code has to be checked against the compiled code.
- **Recomputing the bounds with measured traffic changes two conclusions:**

| Kernel | Bound from model | Bound from counters | Main limiter |
|---|---|---|---|
| smem_tiled | 90% | 73% | shared memory |
| 2d_blocktiling | 45% (compute) | 62% (shared memory) | bank conflicts, 25% occupancy |
| flash_v1 | 90% | 46% | 12% occupancy |
| flash_v2 | 69% | 65% | 25% occupancy |

- **2d_blocktiling is shared-memory-bound, not compute-bound.** 33.6 million bank conflicts (about a third of its shared-memory accesses) raise its traffic to 1.49x the model, which makes shared memory, not arithmetic, its limit.
- **flash_v1 looked efficient only because the model overcounted its traffic.** With the real traffic it reaches 46% of its bound; with 64-thread blocks and 184 registers per thread, only 12% of warp slots are occupied, too few to hide latency.
- **flash_v2's model was accurate (0.94x).** Its remaining gap matches its occupancy: 49.6 KB of shared memory per block allows only one block (8 warps) per SM.
- **Fusion cuts DRAM traffic 35x.** flash_v2 moved 44 MB to and from DRAM against 1.55 GB for unfused attention, close to its compulsory minimum (1.30x).
- **The GEMM DRAM model is far too low, and it does not matter.** At N = 2048 each matrix (16 MB) is larger than L2, so tiles are re-read from DRAM (4.5-45x the compulsory traffic), but that traffic still takes only a fraction of the time spent in L1/shared memory.
- **Small surprises:** softmax executes 2.7x the FLOPs estimated, because `expf` is implemented with several FP32 instructions. The bandwidth-bound kernels move about 15% more DRAM data than the minimum. ECC is enabled on this T4 (`nvidia-smi -q`); on GDDR6 the ECC check data is stored in DRAM and costs about 1/8 of the bandwidth, which is consistent with the extra traffic, although the counters cannot separate it.

## Using the counters to fix a kernel

The counters pointed at two problems in the register-blocked GEMM: 33.6 million bank conflicts and 25% occupancy. Each fix was added as a separate kernel so its effect could be measured on its own (GEMM N = 2048 counters, timings at N = 4096):

| Kernel | Bank conflicts | Shared traffic | Occupancy | GFLOP/s | vs cuBLAS |
|---|---|---|---|---|---|
| 2d_blocktiling | 33.6M | 12.8 GB | 25% | 3,388 | 81.5% |
| 2d_blocktiling_occ (`__launch_bounds__(256, 2)`) | 33.6M | 12.8 GB | 47% | 3,469 | 83.5% |
| 2d_blocktiling_v2 (conflict-free 128-bit loads) | 0.5M | 8.5 GB | 47% | **4,170** | **100.3%** |

- **Occupancy alone helped little (2-7%).** Doubling the warps per SM did not matter while every shared-memory read was still serialised by conflicts.
- **Removing the conflicts was the real fix (23-38% faster).** Storing the A tile transposed and splitting each thread's B columns into two groups of 4 turned every read into a conflict-free 128-bit load. Shared-memory traffic now matches the model (0.99x), the kernel is compute-bound rather than shared-memory-bound, and it matches cuBLAS from N = 1024 to 4096 (on square matrices with dimensions divisible by 128; cuBLAS handles any shape).
- This confirms the Part 3 diagnosis: the counters said shared memory was the limit, and removing shared-memory work was what made the kernel faster.

## Conclusions

1. Measured ceilings, not datasheet numbers: the T4 sustains 92% of its peak FLOP rate and 75-87% of its peak DRAM bandwidth, and its shared memory runs at 64 bytes per clock per SM.
2. A roofline model built from simple counts predicted the memory-bound kernels and the simple GEMMs within 2-10%.
3. Hardware counters were needed to explain the fastest kernels: compiler-merged loads, bank conflicts and occupancy, none of which are visible in the source code.
4. Acting on the counters made the register-blocked GEMM 23% faster at N = 4096, matching cuBLAS.

## How to run (Google Colab, T4 GPU)

Parts 1 and 2 run on their own. Part 3 profiles the kernels from [cuda-transformer-kernels](https://github.com/Synide586/CUDA-Kernals-for-Transformer-ops), so clone both repositories side by side:

```bash
!git clone https:https://github.com/Synide586/CUDA-Kernals-for-Transformer-ops
!git clone https:https://github.com/Synide586/GPU-Roofline-Performance-Model
!cd cuda-transformer-kernels && make
%cd gpu-roofline
```

Then:

```bash
!make ceilings    # Part 1: results/ceilings.csv, results/bandwidth_sweep.png
!make model       # Part 2: results/model_vs_measured.csv, results/roofline.png
!make counters    # Part 3: needs ../cuda-transformer-kernels built with make
```

## Roadmap

- [x] Part 1: measure compute and memory ceilings
- [x] Part 2: analytical model (FLOPs, bytes and predicted time per operator)
- [x] Part 3: validate the model with Nsight Compute hardware counters
