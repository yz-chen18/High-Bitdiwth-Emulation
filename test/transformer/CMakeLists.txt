add_executable(main
  transformerTest.cpp
  ../../random_sample.cpp
  ../../transformer.cpp
  ../../error.cpp
)

target_include_directories(
  main
  PRIVATE
  ../../
)