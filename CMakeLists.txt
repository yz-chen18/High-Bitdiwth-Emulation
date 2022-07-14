find_package(CUDA)

set(CMAKE_C_FLAGS "-O0 -g")
set(CMAKE_CXX_FLAGS "-O0 -g")

cuda_add_executable(main
    main.cu
    transformer.cpp
    error.cpp
)

target_include_directories(main
    PRIVATE
    ./
)

target_link_libraries(main
    PUBLIC
    cublas
)