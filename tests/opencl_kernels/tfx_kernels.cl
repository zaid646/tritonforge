// Portable OpenCL C port of the two fused TritonForge-X kernels.
// This port is used by the CPU-only CI parity test (POCL). The performance
// numbers in results/x/sycl.json come from the SYCL build in sycl/tfx_sycl.cpp.

__kernel void swiglu(__global const float* gate, __global const float* up,
                     __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float x = gate[i];
  out[i] = (x / (1.0f + exp(-x))) * up[i];
}

__kernel void rmsnorm_residual(__global const float* x, __global const float* residual,
                               __global const float* weight, __global float* out,
                               const int cols, const float eps) {
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  const int wg = get_local_size(0);
  __local float scratch[256];

  float local_sum = 0.0f;
  for (int j = lid; j < cols; j += wg) {
    const float s = x[row * cols + j] + residual[row * cols + j];
    local_sum += s * s;
  }
  scratch[lid] = local_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = wg / 2; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float scale = rsqrt(scratch[0] / (float)cols + eps);
  for (int j = lid; j < cols; j += wg) {
    const int idx = row * cols + j;
    out[idx] = (x[idx] + residual[idx]) * scale * weight[j];
  }
}
