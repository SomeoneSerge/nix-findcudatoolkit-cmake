cmake_minimum_required(VERSION 3.25)
project(nix-findacudatoolkit LANGUAGES CXX CUDA)

add_executable(demo main.cc softmax.cu)
target_link_libraries(demo cublas) # Just to see if discovery works
target_compile_features(demo PRIVATE cxx_std_17)
target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                    --expt-relaxed-constexpr>)

install(TARGETS demo)
