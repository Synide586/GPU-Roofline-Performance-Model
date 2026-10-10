#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), \
              __FILE__, __LINE__);                                            \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// median of several timed launches: robust to one-off clock or scheduling hiccups
template <typename Launch>
float median_ms(Launch launch, int reps = 11) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> t(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}

inline int fp32_cores_per_sm(int major, int minor) {
  if (major == 6) return minor == 0 ? 64 : 128;
  if (major == 7) return 64;
  if (major == 8) return minor == 0 ? 64 : 128;
  if (major >= 9) return 128;
  return 64;
}

struct DeviceInfo {
  char name[256];
  int sms, cc_major, cc_minor, cores_per_sm;
  double boost_clock_ghz;
  double peak_gflops;
  double peak_dram_gbs;
  double peak_smem_gbs;
  int l2_bytes;
};

inline DeviceInfo query_device(int dev = 0) {
  DeviceInfo d{};
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.sms = p.multiProcessorCount;
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  d.cores_per_sm = fp32_cores_per_sm(p.major, p.minor);
  d.l2_bytes = p.l2CacheSize;
  int clk_khz = 0, mem_clk_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  d.boost_clock_ghz = clk_khz / 1e6;
  d.peak_gflops = 2.0 * d.sms * d.cores_per_sm * d.boost_clock_ghz;
  d.peak_dram_gbs = 2.0 * (mem_clk_khz * 1e3) * (bus_bits / 8.0) / 1e9;
  // shared memory per SM per clock: 64 bytes on Turing (sm_75), 128 on Volta and Ampere
  const int smem_bytes_per_clk = (p.major == 7 && p.minor == 5) ? 64 : 128;
  d.peak_smem_gbs = d.sms * smem_bytes_per_clk * d.boost_clock_ghz;
  return d;
}
