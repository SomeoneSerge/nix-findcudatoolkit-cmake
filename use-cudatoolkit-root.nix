{ cmake
, cppzmq
, cudaPackages
, glibc
, libsodium
, patchelf
, stdenv
, zeromq
, lddHook
}:

with cudaPackages;

let
  cudaPaths = [
    cuda_nvcc
    cuda_cudart
    cuda_nvtx # It seems that CUDA::cudart won't work without this
    libcublas # Just to see if we can target_link_libraries
  ];
  CUDAToolkit_ROOT = builtins.concatStringsSep ";" (map lib.getDev cudaPaths);
  CUDAToolkit_INCLUDE_DIR = builtins.concatStringsSep ";" (lib.pipe cudaPaths [
    (map lib.getDev)
    (map (x: "${x}/include"))
  ]);
in
stdenv.mkDerivation {
  pname = "demo";
  version = "0.0.1";

  src = ./.;

  nativeBuildInputs = [
    autoAddOpenGLRunpathHook
    cmake
    cuda_nvcc
    lddHook
  ];
  buildInputs = [
    libsodium
    cppzmq
    zeromq
  ] ++ cudaPaths;
  cmakeFlags = [
    "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
    "-DCUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}"
    "-DCMAKE_VERBOSE_MAKEFILE=ON"
    "-DCMAKE_MESSAGE_LOG_LEVEL=TRACE"
  ];

  preConfigure = ''
    export NVCC_APPEND_FLAGS+=" -L${cuda_cudart}/lib -I${cuda_cudart}/include"
  '';

  passthru = { inherit CUDAToolkit_ROOT; };
}

