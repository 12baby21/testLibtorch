cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

set(Torch_DIR /home/wjf/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# main.cpp
add_executable(example gradComputation.cpp example.cpp)

# include
include_directories(./)

# link libtorch .a .so

target_link_libraries(example "${TORCH_LIBRARIES}")

#
set_property(TARGET example PROPERTY CXX_STANDARD 14)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)