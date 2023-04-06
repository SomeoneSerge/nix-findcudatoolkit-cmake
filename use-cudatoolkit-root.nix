{ cmake
, cudaPackages
, glibc
, patchelf
, stdenv
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
  cmakeFlags = [
    "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
    "-DCUDAToolkit_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIR}"
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
  ];

  preConfigure = ''
    echo Environment variables begin >&2
    env >&2
    echo Environment variables end >&2
    echo >&2

    export NVCC_APPEND_FLAGS+=" -L${cuda_cudart}/lib -I${cuda_cudart}/include"
  '';

  nativeCheckInputs = map lib.getBin [
    glibc # ldd
    patchelf
  ];

  doInstallCheck = true;
  preInstallCheck = ''
    if ldd $out/bin/demo | grep -q "not found"
    then
      echo Runpath:
      patchelf --print-rpath $out/bin/demo

      echo DT_NEEDED:
      patchelf --print-needed $out/bin/demo

      echo ldd:
      ldd $out/bin/demo

      exit 1
    fi
  '';

  passthru = { inherit CUDAToolkit_ROOT; };
}

