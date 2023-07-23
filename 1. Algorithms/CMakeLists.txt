cmake_minimum_required(VERSION 3.20)
project(YSDA_Algorithms)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined -fno-sanitize-recover=all -fsanitize-undefined-trap-on-error"
        CACHE STRING "Compiler flags in asan build"
        FORCE)

add_executable(YSDA_Algorithms <task-filename>.cpp)

# ------ run commands ------
# g++-11 S_B_segmentssum.cpp -std=c++17 -O2 -Wall -Werror -Wsign-compare -o fast_solution
# g++-11 S_B_segmentssum.cpp -fsanitize=address,undefined -fno-sanitize-recover=all -std=c++17 -O2 -Wall -Werror -Wsign-compare -o debug_solution
# clang-format -i <task-filename>.cpp
# clang-tidy <task-filename>.cpp -- -std=c++17 -stdlib=libstdc++ -stdlib=libc++
