cmake_minimum_required(VERSION 3.25)
project(nix-findacudatoolkit LANGUAGES CXX CUDA)

find_package(cppzmq REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)

add_executable(demo main.cc softmax.cu)
target_link_libraries(demo PUBLIC CUDA::cublas cppzmq) # Just to see if discovery works
target_compile_features(demo PRIVATE cxx_std_17)
target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                    --expt-relaxed-constexpr>)

install(TARGETS demo)
