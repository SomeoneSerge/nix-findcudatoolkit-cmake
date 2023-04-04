#include <cuda.h>
#include <cuda_runtime.h>

#include "cusoftmax.h"

void cudaFreeOrAbort(void *ptr) {
  if (cudaFree(ptr) != cudaSuccess) {
    std::abort();
  }
}

template <typename Scalar, int maxChannels = 256>
__device__ __forceinline__ Scalar computeDot(const Scalar *f1, const Scalar *f2,
                                             int nChannels) {

  Scalar s_ij = 0;
  for (auto k = 0; k < nChannels; ++k) {
    s_ij += f1[k] * f2[k];
  }

  return s_ij;
}

/* Masked partition function */

template <typename Scalar, int maxChannels = 256>
__global__ void
partitionMaskedCudaForwardKernel(const Scalar *f1, const uint8_t *m1,
                                 const Scalar *f2, const uint8_t *m2,
                                 Scalar *offsets, Scalar *normalizingConstants,
                                 long nPoints1, long nPoints2, int nChannels) {
  const auto b = blockIdx.y;
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  const bool rowActive = i < nPoints1 && m1[b * nPoints1 + i];

  nChannels = std::min(maxChannels, nChannels);

  /* We'll use shared memory to copy f2 */
  __shared__ Scalar shmem[maxChannels];

  /* Let's hope it's optimized out */
  const auto loadF2 = [&](const long j) {
    __syncthreads();
    const auto k = threadIdx.x;
    if (k < nChannels) {
      shmem[k] = f2[b * (nPoints2 * nChannels) + j * nChannels + k];
    } else if (k < maxChannels) {
      shmem[k] = 0;
    }
    __syncthreads();
  };

  /* Copy current f1 to the stack */
  Scalar f1Single[maxChannels];
  for (auto k = 0; k < nChannels; ++k) {
    if (rowActive) {
      f1Single[k] = f1[b * (nPoints1 * nChannels) + i * nChannels + k];
    } else {
      f1Single[k] = 0.0;
    }
  }

  /* The offset, subtracted before exp */
  Scalar maxDot = -std::numeric_limits<Scalar>::infinity();
  for (auto j = 0; j < nPoints2; ++j) {
    loadF2(j);
    Scalar s_ij = computeDot<Scalar, maxChannels>(f1Single, shmem, nChannels);

    const bool colActive = m2[b * nPoints2 + j];
    const bool active = rowActive && colActive;
    if (active && s_ij > maxDot) {
      maxDot = s_ij;
    }
  }

  Scalar z = 0.0;
  for (auto j = 0; j < nPoints2; ++j) {
    loadF2(j);
    const auto add = exp(
        computeDot<Scalar, maxChannels>(f1Single, shmem, nChannels) - maxDot);

    const bool colActive = m2[b * nPoints2 + j];
    const bool active = rowActive && colActive;
    if (active) {
      z += add;
    }
  }

  if (i < nPoints1) {
    normalizingConstants[b * nPoints1 + i] = z;
    offsets[b * nPoints1 + i] = maxDot;
  }
}

template <typename Scalar>
std::tuple<DeviceArray1<Scalar>, DeviceArray1<Scalar>>
partitionMaskedCudaForwardImpl(const DeviceArray2<Scalar> &f1,
                               const DeviceArray1<uint8_t> &m1,
                               const DeviceArray2<Scalar> &f2,
                               const DeviceArray1<uint8_t> &m2) {
  const auto points1 = f1.rows;
  const auto channels = f1.cols;
  const auto points2 = f2.rows;

  DeviceArray1<Scalar> m = cudaMakeUnique<Scalar>(points1);
  DeviceArray1<Scalar> z = cudaMakeUnique<Scalar>(points1);

  constexpr auto maxChannels = 256;
  constexpr auto threads = 256;
  dim3 blocks((points1 + threads - 1) / threads, 1);

  partitionMaskedCudaForwardKernel<Scalar, maxChannels>
      <<<blocks, threads>>>(f1.get(), m1.get(), f2.get(), m2.get(), m.get(),
                            z.get(), points1, points2, channels);

  return std::make_tuple(std::move(m), std::move(z));
}

std::tuple<DeviceArray1<float>, DeviceArray1<float>> partitionMaskedCudaForward(
    const DeviceArray2<float> &f1, const DeviceArray1<uint8_t> &m1,
    const DeviceArray2<float> &f2, const DeviceArray1<uint8_t> &m2) {
  return partitionMaskedCudaForwardImpl<float>(f1, m1, f2, m2);
}
