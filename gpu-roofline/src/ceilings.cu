#include <cstring>
#include <string>

#include "common.cuh"

constexpr int THREADS = 256;

// 8 independent FMA chains per thread so the pipeline never waits on a previous result
template <int ILP>
__global__ void fma_peak(float* out, int iters, float a, float b) {
  float x[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) x[i] = threadIdx.x * 1e-3f + i;
#pragma unroll 4
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) x[i] = fmaf(x[i], a, b);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < ILP; ++i) s += x[i];
  if (s == -1.f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// __ldcg bypasses L1; launched with only resident blocks so every pass sweeps the whole buffer
__global__ void read_kernel(const float4* __restrict__ a, size_t n4, int passes, float* out) {
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; ++p)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
      const float4 v = __ldcg(a + i);
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  if (acc.x + acc.y + acc.z + acc.w == -1.f) out[0] = acc.x;
}

__global__ void write_kernel(float4* __restrict__ a, size_t n4, float v) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
    a[i] = make_float4(v, v, v, v);
}

__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n4) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) b[i] = a[i];
}

__global__ void triad_kernel(float4* __restrict__ a, const float4* __restrict__ b,
                             const float4* __restrict__ c, size_t n4, float s) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 x = b[i], y = c[i];
    a[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
  }
}

// 128-bit conflict-free loads; unroll 1 and a non-repeating stride stop the compiler merging repeated loads
__global__ void smem_read(float* out, int iters) {
  constexpr int SIZE = 512;
  __shared__ float4 buf[SIZE];
  for (int i = threadIdx.x; i < SIZE; i += blockDim.x) buf[i] = make_float4(i, i + 1, i + 2, i + 3);
  __syncthreads();
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
  const int t = threadIdx.x;
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const int o = (it * 32) & (SIZE - 1);
    const float4 x = buf[(t + o) & (SIZE - 1)];
    const float4 y = buf[(t + o + 256) & (SIZE - 1)];
    a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
    b.x += y.x; b.y += y.y; b.z += y.z; b.w += y.w;
  }
  if (a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w == -1.f) out[blockIdx.x * blockDim.x + threadIdx.x] = a.x;
}

struct Row {
  std::string name, unit;
  double measured, theoretical;
};

int main(int argc, char** argv) {
  std::string out_dir = "results";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    else {
      fprintf(stderr, "usage: %s [--out results]\n", argv[0]);
      return 1;
    }
  }

  const DeviceInfo dev = query_device();
  printf("Device: %s | sm_%d%d | %d SMs | %d FP32 cores/SM | boost clock %.2f GHz | L2 %d KB\n\n",
         dev.name, dev.cc_major, dev.cc_minor, dev.sms, dev.cores_per_sm, dev.boost_clock_ghz,
         dev.l2_bytes / 1024);

  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)dev.sms * 64 * THREADS * sizeof(float)));
  std::vector<Row> rows;

  // ---------------- FP32 compute ----------------
  {
    constexpr int ILP = 8;
    const int blocks = dev.sms * 32, iters = 1 << 17;
    const double flops = 2.0 * ILP * (double)iters * blocks * THREADS;
    const float ms = median_ms([&] { fma_peak<ILP><<<blocks, THREADS>>>(d_out, iters, 0.9999f, 1e-4f); });
    const double gflops = flops / (ms * 1e6);
    rows.push_back({"fp32_fma", "GFLOP/s", gflops, dev.peak_gflops});
    printf("FP32 FMA throughput : %8.0f GFLOP/s  (theoretical %.0f, %.0f%%)\n", gflops,
           dev.peak_gflops, 100.0 * gflops / dev.peak_gflops);
    const double implied_clock = gflops / (2.0 * dev.sms * dev.cores_per_sm);
    printf("  implied sustained SM clock: %.2f GHz (boost %.2f GHz)\n\n", implied_clock, dev.boost_clock_ghz);
    rows.push_back({"implied_sm_clock", "GHz", implied_clock, dev.boost_clock_ghz});
  }

  // ---------------- DRAM bandwidth ----------------
  {
    const size_t n = (size_t)64 << 20;  // 64M floats = 256 MB per buffer, far larger than L2
    const size_t n4 = n / 4, bytes = n * sizeof(float);
    float4 *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    const int blocks = dev.sms * 16;
    write_kernel<<<blocks, THREADS>>>(a, n4, 1.f);
    write_kernel<<<blocks, THREADS>>>(b, n4, 2.f);
    write_kernel<<<blocks, THREADS>>>(c, n4, 3.f);
    CUDA_CHECK(cudaDeviceSynchronize());

    const float ms_read = median_ms([&] { read_kernel<<<blocks, THREADS>>>(a, n4, 1, d_out); });
    const float ms_write = median_ms([&] { write_kernel<<<blocks, THREADS>>>(a, n4, 0.5f); });
    const float ms_copy = median_ms([&] { copy_kernel<<<blocks, THREADS>>>(b, a, n4); });
    const float ms_triad = median_ms([&] { triad_kernel<<<blocks, THREADS>>>(a, b, c, n4, 0.5f); });
    const double gbs_read = bytes / (ms_read * 1e6);
    const double gbs_write = bytes / (ms_write * 1e6);
    const double gbs_copy = 2.0 * bytes / (ms_copy * 1e6);
    const double gbs_triad = 3.0 * bytes / (ms_triad * 1e6);
    printf("DRAM bandwidth (256 MB buffers, theoretical %.0f GB/s)\n", dev.peak_dram_gbs);
    printf("  read  : %7.1f GB/s (%.0f%%)\n", gbs_read, 100.0 * gbs_read / dev.peak_dram_gbs);
    printf("  write : %7.1f GB/s (%.0f%%)\n", gbs_write, 100.0 * gbs_write / dev.peak_dram_gbs);
    printf("  copy  : %7.1f GB/s (%.0f%%)\n", gbs_copy, 100.0 * gbs_copy / dev.peak_dram_gbs);
    printf("  triad : %7.1f GB/s (%.0f%%)\n\n", gbs_triad, 100.0 * gbs_triad / dev.peak_dram_gbs);
    rows.push_back({"dram_read", "GB/s", gbs_read, dev.peak_dram_gbs});
    rows.push_back({"dram_write", "GB/s", gbs_write, dev.peak_dram_gbs});
    rows.push_back({"dram_copy", "GB/s", gbs_copy, dev.peak_dram_gbs});
    rows.push_back({"dram_triad", "GB/s", gbs_triad, dev.peak_dram_gbs});
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    // ---------------- working-set sweep: L2 vs DRAM ----------------
    const std::string sweep_path = out_dir + "/bandwidth_sweep.csv";
    FILE* f = fopen(sweep_path.c_str(), "w");
    if (!f) { perror("bandwidth_sweep.csv"); return 1; }
    fprintf(f, "bytes,gbs\n");
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, read_kernel, THREADS, 0));
    const int resident_blocks = per_sm * dev.sms;
    printf("Read bandwidth vs working-set size (L2 = %d KB, %d resident blocks)\n", dev.l2_bytes / 1024,
           resident_blocks);
    double best_l2 = 0.0;
    const size_t kb_sizes[] = {512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384, 32768, 65536, 131072, 262144};
    for (size_t kb : kb_sizes) {
      const size_t sz = kb * 1024, s4 = sz / 16;
      const int passes = (int)std::max<size_t>(1, ((size_t)2 << 30) / sz);  // about 2 GB read per launch
      const int sblocks = (int)std::min<size_t>((s4 + THREADS - 1) / THREADS, (size_t)resident_blocks);
      const float ms = median_ms([&] { read_kernel<<<sblocks, THREADS>>>(a, s4, passes, d_out); }, 5);
      const double gbs = (double)sz * passes / (ms * 1e6);
      if (sz <= (size_t)dev.l2_bytes / 2) best_l2 = std::max(best_l2, gbs);
      printf("  %8.1f MB : %7.1f GB/s\n", sz / 1048576.0, gbs);
      fprintf(f, "%zu,%.2f\n", sz, gbs);
    }
    fclose(f);
    printf("\n");
    rows.push_back({"l2_read", "GB/s", best_l2, 0.0});
    CUDA_CHECK(cudaFree(a));
  }

  // ---------------- shared-memory bandwidth ----------------
  {
    const int blocks = dev.sms * 8, iters = 1 << 17;
    const double traffic = 2.0 * sizeof(float4) * (double)iters * blocks * THREADS;
    const float ms = median_ms([&] { smem_read<<<blocks, THREADS>>>(d_out, iters); });
    const double gbs = traffic / (ms * 1e6);
    printf("Shared-memory read : %8.0f GB/s  (theoretical %.0f, %.0f%%)\n\n", gbs, dev.peak_smem_gbs,
           100.0 * gbs / dev.peak_smem_gbs);
    rows.push_back({"smem_read", "GB/s", gbs, dev.peak_smem_gbs});
  }

  const std::string path = out_dir + "/ceilings.csv";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { perror("ceilings.csv"); return 1; }
  fprintf(f, "name,unit,measured,theoretical\n");
  for (auto& r : rows) fprintf(f, "%s,%s,%.2f,%.2f\n", r.name.c_str(), r.unit.c_str(), r.measured, r.theoretical);
  fclose(f);
  printf("Wrote %s and %s/bandwidth_sweep.csv\n", path.c_str(), out_dir.c_str());

  const double ridge = rows[0].measured / rows[2].measured;
  printf("Ridge point (measured FP32 / DRAM read): %.1f FLOP/byte\n", ridge);
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
