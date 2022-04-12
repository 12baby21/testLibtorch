cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example LANGUAGES CXX)

# main.cpp
aux_source_directory(./src SOURCE_DIR)
add_executable(example example.cpp ${SOURCE_DIR})
include_directories(./include)
add_subdirectory(hard_api)


# link libtorch .a .so
target_link_libraries(example gmp pthread)
target_link_libraries(example paillier_api)

set_property(TARGET example PROPERTY CXX_STANDARD 14)
