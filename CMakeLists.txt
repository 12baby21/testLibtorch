cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example LANGUAGES CXX)

# Libtorch directory
set(Torch_DIR /home/wjf/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Disable C++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)
# Require full C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# main.cpp
aux_source_directory(./ SOURCE_DIR)
add_executable(example ${SOURCE_DIR})

# link libtorch .a .so
target_link_libraries(example "${TORCH_LIBRARIES}")
target_link_libraries(example gmp)

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