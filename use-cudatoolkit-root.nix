{ cmake
, cppzmq
, cudaPackages
, glibc
, patchelf
, stdenv
, zeromq
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
  ];
  buildInputs = [
    cppzmq
    zeromq
  ];
  cmakeFlags = [
    "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
    "-DCUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}"
    "-DCMAKE_VERBOSE_MAKEFILE=ON"
    "-DCMAKE_MESSAGE_LOG_LEVEL=TRACE"
  ];

  preConfigure = ''
    export NVCC_APPEND_FLAGS+=" -L${cuda_cudart}/lib -I${cuda_cudart}/include"
  '';

  nativeCheckInputs = map lib.getBin [
    glibc # ldd
    patchelf
  ];

  doInstallCheck = true;
  preInstallCheck = ''
    echo ldd $(pwd)/demo
    ldd $(pwd)/demo

    if ldd $out/bin/demo | grep -q "not found"
    then
      echo patchelf --print-rpath $out/bin/demo
      patchelf --print-rpath $out/bin/demo
      echo

      echo patchelf --print-needed $out/bin/demo
      patchelf --print-needed $out/bin/demo
      echo

      echo ldd $out/bin/demo
      ldd $out/bin/demo

      # exit 1
    fi
  '';

  passthru = { inherit CUDAToolkit_ROOT; };
}

