{ stdenv
, cmake
, cudaPackages
}:

with cudaPackages;

let
  cudaPaths = [
    cuda_nvcc
    cuda_cudart
    libcublas # Just to see if we can target_link_libraries
  ];
  CUDAToolkit_ROOT = builtins.concatStringsSep ";" cudaPaths;
in
stdenv.mkDerivation {
  pname = "demo";
  version = "0.0.1";

  src = ./.;

  nativeBuildInputs = [
    cmake
    cuda_nvcc
  ];
  cmakeFlags = [
    "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
  ];

  preConfigure = ''
    echo Environment variables begin >&2
    env >&2
    echo Environment variables end >&2
    echo >&2
  '';

  passthru = { inherit CUDAToolkit_ROOT; };
}

