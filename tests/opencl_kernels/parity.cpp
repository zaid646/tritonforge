// Minimal OpenCL C host used only by the CPU-only CI parity test.
// Reads raw float32 inputs, runs the kernel from tfx_kernels.cl, writes raw
// float32 output. No timing and no performance claims are made here.

#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

void write_f32(const std::string& path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
}

void check(cl_int status, const char* what) {
  if (status != CL_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", what, status);
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel_name = "swiglu", source, in0, in1, in2, out_path;
  int m = 1, n = 896, wg = 128;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) std::exit(2);
      return argv[++i];
    };
    if (key == "--kernel") kernel_name = next();
    else if (key == "--source") source = next();
    else if (key == "--m") m = std::stoi(next());
    else if (key == "--n") n = std::stoi(next());
    else if (key == "--wg") wg = std::stoi(next());
    else if (key == "--in0") in0 = next();
    else if (key == "--in1") in1 = next();
    else if (key == "--in2") in2 = next();
    else if (key == "--out") out_path = next();
  }
  if (source.empty() || in0.empty() || in1.empty() || out_path.empty()) {
    std::fprintf(stderr, "usage: %s --source kernels.cl --kernel swiglu|rmsnorm "
                         "--in0 a --in1 b [--in2 w] --out y --m --n --wg\n", argv[0]);
    return 2;
  }
  std::ifstream src_file(source);
  const std::string src((std::istreambuf_iterator<char>(src_file)),
                        std::istreambuf_iterator<char>());
  if (src.empty()) {
    std::fprintf(stderr, "empty kernel source: %s\n", source.c_str());
    return 1;
  }

  cl_uint platform_count = 0;
  check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
  if (platform_count == 0) {
    std::fprintf(stderr, "no OpenCL platform\n");
    return 3;
  }
  std::vector<cl_platform_id> platforms(platform_count);
  check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
  cl_platform_id platform = platforms[0];
  cl_device_id device = nullptr;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr) != CL_SUCCESS) {
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr), "clGetDeviceIDs");
  }

  cl_int err = CL_SUCCESS;
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  check(err, "clCreateContext");
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  check(err, "clCreateCommandQueue");
  const char* src_ptr = src.c_str();
  const size_t src_len = src.size();
  cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
  check(err, "clCreateProgramWithSource");
  err = clBuildProgram(program, 1, &device, "-cl-mad-enable", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::fprintf(stderr, "build log:\n%s\n", log.data());
    return 1;
  }
  cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
  check(err, "clCreateKernel");

  std::vector<float> a = read_f32(in0);
  std::vector<float> b = read_f32(in1);
  std::vector<float> w = in2.empty() ? std::vector<float>() : read_f32(in2);
  std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);

  cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                a.size() * sizeof(float), a.data(), &err);
  check(err, "buffer a");
  cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                b.size() * sizeof(float), b.data(), &err);
  check(err, "buffer b");
  cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out.size() * sizeof(float), nullptr,
                                  &err);
  check(err, "buffer out");
  cl_mem buf_w = nullptr;
  if (!w.empty()) {
    buf_w = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           w.size() * sizeof(float), w.data(), &err);
    check(err, "buffer w");
  }

  if (kernel_name == "swiglu") {
    const int total = m * n;
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a), "arg0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b), "arg1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out), "arg2");
    check(clSetKernelArg(kernel, 3, sizeof(int), &total), "arg3");
    const size_t global = ((total + wg - 1) / wg) * wg;
    const size_t local = wg;
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue swiglu");
  } else {
    const float eps = 1e-6f;
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a), "arg0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b), "arg1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_w), "arg2");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_out), "arg3");
    check(clSetKernelArg(kernel, 4, sizeof(int), &n), "arg4");
    check(clSetKernelArg(kernel, 5, sizeof(float), &eps), "arg5");
    const size_t local = wg;
    const size_t global = static_cast<size_t>(m) * wg;
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "enqueue rmsnorm");
  }
  check(clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, out.size() * sizeof(float), out.data(), 0,
                            nullptr, nullptr),
        "read");
  write_f32(out_path, out);

  clReleaseMemObject(buf_a);
  clReleaseMemObject(buf_b);
  if (buf_w) clReleaseMemObject(buf_w);
  clReleaseMemObject(buf_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}
