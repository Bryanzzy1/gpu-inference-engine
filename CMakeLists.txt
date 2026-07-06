cmake_minimum_required(VERSION 3.16)
project(gpu_inference_trading LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header search path.
include_directories(${CMAKE_SOURCE_DIR}/include)

# Toolchain smoke test.
add_executable(hello src/hello.cpp)

# Tick parser.
add_executable(parse_trades src/main.cpp src/parser.cpp)
target_compile_features(parse_trades PRIVATE cxx_std_17)

# Feature + label CSV builder.
add_executable(build_features src/build_features.cpp src/parser.cpp src/features.cpp)
target_compile_features(build_features PRIVATE cxx_std_17)
