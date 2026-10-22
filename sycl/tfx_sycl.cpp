// TritonForge-X: fused SwiGLU, fused RMSNorm+residual, and a STREAM-style
// bandwidth probe for the CPU SYCL device.
//
// Build (Intel oneAPI DPC++):
//   icpx -fsycl -O3 -std=c++17 -march=native sycl/tfx_sycl.cpp -o build/tfx_sycl
//
// Run on the CPU device:
//   ONEAPI_DEVICE_SELECTOR=*:cpu build/tfx_sycl --op swiglu --m 512 --n 4864 --wg 256

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string op = "info";
  size_t m = 512;
  size_t n = 4864;
  size_t wg = 256;
  size_t vec = 4;
  int iters = 50;
  int warmup = 5;
  std::string json;
  std::string dump;
  size_t mb = 128;
  uint32_t seed = 0;
};

double percentile(std::vector<double> values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double index = (values.size() - 1) * q;
  const size_t low = static_cast<size_t>(index);
  const size_t high = std::min(low + 1, values.size() - 1);
  return values[low] * (1 - (index - low)) + values[high] * (index - low);
}

void dump_f32(const std::string& path, const float* data, size_t count) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

void write_json(const Args& args, const std::string& device, double ms_mean, double ms_p95,
                double gbps, double gflops) {
  if (args.json.empty()) return;
  std::ofstream out(args.json);
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  out << "  \"op\": \"" << args.op << "\",\n";
  out << "  \"device\": \"" << device << "\",\n";
  out << "  \"m\": " << args.m << ",\n";
  out << "  \"n\": " << args.n << ",\n";
  out << "  \"work_group\": " << args.wg << ",\n";
  out << "  \"vector_width\": " << args.vec << ",\n";
  out << "  \"iters\": " << args.iters << ",\n";
  out << "  \"ms_mean\": " << ms_mean << ",\n";
  out << "  \"ms_p95\": " << ms_p95 << ",\n";
  out << "  \"gbps\": " << gbps << ",\n";
  out << "  \"gflops\": " << gflops << "\n";
  out << "}\n";
}

sycl::event sycl_swiglu(sycl::queue& q, float* gate, float* up, float* out, size_t n, size_t wg,
               size_t vec) {
  const size_t threads = wg;
  const bool use_vec4 = (vec == 4 && n % 4 == 0);
  const size_t per = use_vec4 ? 4 : 1;
  const size_t groups = (n / per + threads - 1) / threads;
  if (use_vec4) {
    auto* g4 = reinterpret_cast<sycl::vec<float, 4>*>(gate);
    auto* u4 = reinterpret_cast<sycl::vec<float, 4>*>(up);
    auto* o4 = reinterpret_cast<sycl::vec<float, 4>*>(out);
    const size_t n4 = n / 4;
    return q.parallel_for(sycl::nd_range<1>(groups * threads, threads), [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      if (i >= n4) return;
      const sycl::vec<float, 4> g = g4[i];
      const sycl::vec<float, 4> u = u4[i];
      sycl::vec<float, 4> o;
      for (int k = 0; k < 4; ++k) {
        const float x = g[k];
        o[k] = (x / (1.0f + sycl::exp(-x))) * u[k];
      }
      o4[i] = o;
    });
  }
  return q.parallel_for(sycl::nd_range<1>(groups * threads, threads), [=](sycl::nd_item<1> it) {
    const size_t i = it.get_global_id(0);
    if (i >= n) return;
    const float x = gate[i];
    out[i] = (x / (1.0f + sycl::exp(-x))) * up[i];
  });
}

sycl::event sycl_rmsnorm_residual(sycl::queue& q, const float* x, const float* residual,
                                  const float* weight, float* out, size_t rows, size_t cols,
                                  size_t wg, float eps = 1e-6f) {
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> summed{sycl::range<1>(cols), h};
    h.parallel_for(sycl::nd_range<1>(rows * wg, wg), [=](sycl::nd_item<1> it) {
      const size_t row = it.get_group(0);
      const size_t lid = it.get_local_id(0);
      const size_t base = row * cols;
      float local_sum = 0.0f;
      for (size_t j = lid; j < cols; j += wg) {
        const float s = x[base + j] + residual[base + j];
        summed[j] = s;
        local_sum += s * s;
      }
      const float total = sycl::reduce_over_group(it.get_group(), local_sum, sycl::plus<float>());
      const float scale = sycl::rsqrt(total / static_cast<float>(cols) + eps);
      for (size_t j = lid; j < cols; j += wg) {
        out[base + j] = summed[j] * scale * weight[j];
      }
    });
  });
}

sycl::event sycl_stream(sycl::queue& q, const float* a, const float* b, float* c, size_t n,
                        size_t wg, size_t vec) {
  const size_t per = std::max<size_t>(vec, 1);
  const size_t groups = (n / per + wg - 1) / wg;
  if (vec == 4 && n % 4 == 0) {
    auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(a);
    auto* b4 = reinterpret_cast<const sycl::vec<float, 4>*>(b);
    auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(c);
    const size_t n4 = n / 4;
    return q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      if (i >= n4) return;
      const sycl::vec<float, 4> av = a4[i];
      const sycl::vec<float, 4> bv = b4[i];
      c4[i] = av + bv;
    });
  }
  return q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {
    const size_t i = it.get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
  });
}

template <typename Kernel>
double time_kernel(sycl::queue& q, const Args& args, Kernel&& launch, size_t bytes,
                   size_t flops, std::vector<double>& samples) {
  for (int i = 0; i < args.warmup; ++i) {
    launch();
  }
  q.wait();
  for (int i = 0; i < args.iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    launch();
    q.wait();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  const size_t trim = samples.size() / 10;
  const double mean =
      std::accumulate(samples.begin() + trim, samples.end() - trim, 0.0) / (samples.size() - 2 * trim);
  const double p95 = percentile(samples, 0.95);
  (void)bytes;
  (void)flops;
  return mean;
}

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << key << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (key == "--op") args.op = value();
    else if (key == "--m") args.m = std::stoul(value());
    else if (key == "--n") args.n = std::stoul(value());
    else if (key == "--wg") args.wg = std::stoul(value());
    else if (key == "--vec") args.vec = std::stoul(value());
    else if (key == "--iters") args.iters = std::stoi(value());
    else if (key == "--warmup") args.warmup = std::stoi(value());
    else if (key == "--json") args.json = value();
    else if (key == "--dump") args.dump = value();
    else if (key == "--mb") args.mb = std::stoul(value());
    else if (key == "--seed") args.seed = std::stoul(value());
    else {
      std::cerr << "unknown argument " << key << "\n";
      std::exit(2);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
  const std::string device = q.get_device().get_info<sycl::info::device::name>();

  if (args.op == "info") {
    std::cout << "device: " << device << "\n";
    std::cout << "vendor: " << q.get_device().get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "driver: " << q.get_device().get_info<sycl::info::device::driver_version>()
              << "\n";
    std::cout << "max_work_group_size: "
              << q.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
    std::cout << "local_mem_kb: "
              << q.get_device().get_info<sycl::info::device::local_mem_size>() / 1024 << "\n";
    return 0;
  }

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  if (args.op == "stream") {
    const size_t n = args.mb * 1024 * 1024 / sizeof(float);
    auto* a = sycl::malloc_shared<float>(n, q);
    auto* b = sycl::malloc_shared<float>(n, q);
    auto* c = sycl::malloc_shared<float>(n, q);
    for (size_t i = 0; i < n; ++i) {
      a[i] = dist(rng);
      b[i] = dist(rng);
    }
    q.wait();
    std::vector<double> samples;
    const size_t bytes = n * 3 * sizeof(float);
    const double mean = time_kernel(
        q, args,
        [&]() { sycl_stream(q, a, b, c, n, args.wg, args.vec); },
        bytes, 0, samples);
    const double p95 = percentile(samples, 0.95);
    const double gbps = bytes / (mean * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(3)
              << "stream mb=" << args.mb << " wg=" << args.wg << " vec=" << args.vec
              << " mean_ms=" << mean << " p95_ms=" << p95 << " gbps=" << gbps << "\n";
    write_json(args, device, mean, p95, gbps, 0.0);
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return 0;
  }

  const size_t elements = args.op == "swiglu" ? args.m * args.n : args.m * args.n;
  auto* a = sycl::malloc_shared<float>(elements, q);
  auto* b = sycl::malloc_shared<float>(elements, q);
  auto* out = sycl::malloc_shared<float>(elements, q);
  auto* weight = sycl::malloc_shared<float>(args.op == "rmsnorm_residual" ? args.n : 1, q);
  for (size_t i = 0; i < elements; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    out[i] = 0.0f;
  }
  for (size_t i = 0; i < (args.op == "rmsnorm_residual" ? args.n : 1); ++i) {
    weight[i] = 0.5f + 0.5f * std::abs(dist(rng));
  }
  q.wait();

  std::vector<double> samples;
  double mean = 0.0;
  size_t bytes = 0;
  size_t flops = 0;
  if (args.op == "swiglu") {
    bytes = elements * 3 * sizeof(float);
    flops = elements * 5;
    mean = time_kernel(
        q, args,
        [&]() { sycl_swiglu(q, a, b, out, elements, args.wg, args.vec); },
        bytes, flops, samples);
  } else if (args.op == "rmsnorm_residual") {
    bytes = elements * 4 * sizeof(float);
    flops = elements * 6;
    mean = time_kernel(
        q, args,
        [&]() {
          sycl_rmsnorm_residual(q, a, b, weight, out, args.m, args.n, args.wg);
        },
        bytes, flops, samples);
  } else {
    std::cerr << "unknown op " << args.op << "\n";
    return 2;
  }

  const double p95 = percentile(samples, 0.95);
  const double gbps = bytes / (mean * 1e-3) / 1e9;
  const double gflops = flops / (mean * 1e-3) / 1e9;
  std::cout << std::fixed << std::setprecision(3)
            << args.op << " m=" << args.m << " n=" << args.n << " wg=" << args.wg
            << " vec=" << args.vec << " mean_ms=" << mean << " p95_ms=" << p95
            << " gbps=" << gbps << " gflops=" << gflops << "\n";

  if (!args.dump.empty()) {
    if (args.op == "swiglu") {
      dump_f32(args.dump + "/gate.bin", a, elements);
      dump_f32(args.dump + "/up.bin", b, elements);
      dump_f32(args.dump + "/out.bin", out, elements);
    } else {
      dump_f32(args.dump + "/x.bin", a, elements);
      dump_f32(args.dump + "/residual.bin", b, elements);
      dump_f32(args.dump + "/weight.bin", weight, args.n);
      dump_f32(args.dump + "/out.bin", out, elements);
    }
  }
  write_json(args, device, mean, p95, gbps, gflops);

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(out, q);
  sycl::free(weight, q);
  return 0;
}
