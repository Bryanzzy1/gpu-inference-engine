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

# Warnings on for the CPU targets. GCC/Clang and MSVC take different flags.
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

# On MinGW, link the GCC runtimes statically. Otherwise the exe loads whichever
# libstdc++-6.dll is first on PATH, and an older one (Git for Windows ships its
# own) makes the binaries segfault on launch.
if(MINGW)
  add_link_options(-static-libstdc++ -static-libgcc -static)
endif()

# Toolchain smoke test.
add_executable(hello src/tools/hello.cpp)

# Tick parser.
add_executable(parse_trades src/io/main.cpp src/io/parser.cpp)
target_compile_features(parse_trades PRIVATE cxx_std_17)

# Feature + label CSV builder.
add_executable(build_features src/io/build_features.cpp src/io/parser.cpp src/io/features.cpp)
target_compile_features(build_features PRIVATE cxx_std_17)

# C++ model forward pass vs the Python reference.
add_executable(match_model src/cpu/match_model.cpp src/cpu/model.cpp)
target_compile_features(match_model PRIVATE cxx_std_17)

# Tail-latency benchmark of the CPU forward pass.
add_executable(bench_model src/bench/bench_model.cpp src/cpu/model.cpp src/io/parser.cpp src/io/features.cpp src/cpu/latency.cpp)
target_compile_features(bench_model PRIVATE cxx_std_17)

# CPU sweep of the 2D frontier (batch x arrival rate). GPU version needs nvcc.
add_executable(frontier_cpu src/bench/frontier_cpu.cpp src/bench/frontier.cpp src/cpu/model.cpp src/io/parser.cpp src/io/features.cpp src/cpu/latency.cpp)
target_compile_features(frontier_cpu PRIVATE cxx_std_17)

# SPSC lock-free ring self-check (one producer thread, one consumer thread).
add_executable(test_ring src/tools/test_ring.cpp)
target_compile_features(test_ring PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(test_ring PRIVATE Threads::Threads)
