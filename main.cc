#include "cusoftmax.h"
#include <iostream>
#include <vector>

int main(int argc, char **argv) {

  // clang-format off
  std::vector<float> f1 = {
      0.25, 0.75, 0.25, 0.00,
      0.00, 0.25, 0.75, 0.25,
      0.00, 0.00, 0.25, 0.75,
  };
  std::vector<float> f2 = {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  };
  // clang-format on

  constexpr auto nChannels = 4;

  auto f1Dev = DeviceArray2<float>(f1.size() / nChannels, nChannels);
  auto f2Dev = DeviceArray2<float>(f2.size() / nChannels, nChannels);

  auto m1Dev = cudaMakeUnique<uint8_t>(f1Dev.rows);
  auto m2Dev = cudaMakeUnique<uint8_t>(f2Dev.rows);

  if (cudaMemcpy(f1Dev.get(), f1.data(), f1.size() * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::abort();
  }
  if (cudaMemcpy(f2Dev.get(), f2.data(), f2.size() * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::abort();
  }
  if (cudaMemset(m1Dev.get(), 1, f1Dev.rows) != cudaSuccess) {
    std::abort();
  }
  if (cudaMemset(m2Dev.get(), 1, f2Dev.rows) != cudaSuccess) {
    std::abort();
  }

  auto [maxDev, partDev] =
      partitionMaskedCudaForward(f1Dev, m1Dev, f2Dev, m2Dev);

  std::vector<float> part(f1Dev.rows);
  std::vector<float> max(f1Dev.rows);

  if (cudaMemcpy(part.data(), partDev.get(), part.size() * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
  }
  if (cudaMemcpy(max.data(), maxDev.get(), max.size() * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
  }

  std::cout << "offset\tz" << std::endl;
  for (auto i = 0; i < part.size(); ++i) {
    std::cout << max[i] << "\t" << part[i] << std::endl;
  }

  return 0;
}
