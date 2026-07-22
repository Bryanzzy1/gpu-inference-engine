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

# On MinGW, link the GCC runtimes statically. Otherwise the exe loads whichever
# libstdc++-6.dll is first on PATH, and an older one (Git for Windows ships its
# own) makes the binaries segfault on launch.
if(MINGW)
  add_link_options(-static-libstdc++ -static-libgcc -static)
endif()

# Toolchain smoke test.
add_executable(hello src/hello.cpp)

# Tick parser.
add_executable(parse_trades src/main.cpp src/parser.cpp)
target_compile_features(parse_trades PRIVATE cxx_std_17)

# Feature + label CSV builder.
add_executable(build_features src/build_features.cpp src/parser.cpp src/features.cpp)
target_compile_features(build_features PRIVATE cxx_std_17)

# C++ model forward pass vs the Python reference.
add_executable(match_model src/match_model.cpp src/model.cpp)
target_compile_features(match_model PRIVATE cxx_std_17)

# Tail-latency benchmark of the CPU forward pass.
add_executable(bench_model src/bench_model.cpp src/model.cpp src/parser.cpp src/features.cpp src/latency.cpp)
target_compile_features(bench_model PRIVATE cxx_std_17)
