{ stdenv
, cmake
, cudaPackages
}:

with cudaPackages;

stdenv.mkDerivation {
  pname = "demo";
  version = "0.0.1";

  src = ./.;

  nativeBuildInputs = [
    cmake
    cuda_nvcc
  ];
  buildInputs = [
    cuda_cudart
    libcublas # Just to see if we can target_link_libraries
  ];
}


