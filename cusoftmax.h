#ifndef KERNEL_H_
#define KERNEL_H_

#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>

void cudaFreeOrAbort(void *ptr);

template <typename Scalar>
using DeviceArray1 = std::unique_ptr<Scalar[], decltype(&cudaFreeOrAbort)>;

template <typename Scalar>
inline DeviceArray1<Scalar> cudaMakeUnique(long nScalars) {
  Scalar *out;
  if (cudaMalloc(&out, nScalars * sizeof(Scalar)) == cudaSuccess) {
    return DeviceArray1<Scalar>(out, cudaFreeOrAbort);
  }
  return DeviceArray1<Scalar>(nullptr, cudaFreeOrAbort);
}

// A row-major matrix stored on the device that has only a single owner
template <typename Scalar> struct DeviceArray2 {
  DeviceArray2(long rows, long cols)
      : rows(rows), cols(cols), data(cudaMakeUnique<Scalar>(rows * cols)) {}

  const Scalar &operator()(long i, long j) const { return data[i * cols + j]; }
  Scalar &operator()(long i, long j) { return data[i * cols + j]; }

  const Scalar *get() const { return data.get(); }
  Scalar *get() { return data.get(); }

  long rows, cols;
  DeviceArray1<Scalar> data;
};

std::tuple<DeviceArray1<float>, DeviceArray1<float>> partitionMaskedCudaForward(
    const DeviceArray2<float> &f1, const DeviceArray1<uint8_t> &m1,
    const DeviceArray2<float> &f2, const DeviceArray1<uint8_t> &m2);

#endif
