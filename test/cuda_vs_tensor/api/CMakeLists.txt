find_package(CUDA)

set(CUDA_NVCC_FLAGS "-std=c++11 --ptxas-options=-v -use_fast_math -ccbin g++ -arch=compute_80 -code=sm_80 -expt-relaxed-constexpr --compiler-options -fPIC -I$(CUDA_SDK_PATH)/Common/  -L$(CUDA_SDK_PATH)/lib")
set(CMAKE_C_FLAGS "-O0 -g")
set(CMAKE_CXX_FLAGS "-O0 -g")

cuda_add_executable(main
    ./main.cu
    ../../../error.cpp
    ../../../transformer.cpp
    ../../../Gemm.cu
    ../../../random_sample.cpp
)

target_include_directories(main
    PRIVATE
    ../../../
)

target_link_libraries(main
    PRIVATE
    cublas
)