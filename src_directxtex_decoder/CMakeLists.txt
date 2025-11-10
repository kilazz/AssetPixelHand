# Minimum required version of CMake.
cmake_minimum_required(VERSION 3.12)

# The project name.
project(directxtex_decoder_project)

# Find the pybind11 package, required for creating Python bindings.
find_package(pybind11 REQUIRED)

# Set the C++17 standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Subproject Build Optimizations ---
# Disable building parts of DirectXTex that we don't need (tools and tests).
set(BUILD_TOOLS OFF CACHE BOOL "Build command line tools" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build unit tests" FORCE)

# Add the DirectXTex library as a subproject from the external folder.
add_subdirectory(external/DirectXTex)

# --- Main Target Definition ---
# Create the Python module "directxtex_decoder" from its source file.
pybind11_add_module(directxtex_decoder directxtex_decoder/decoder.cpp)

# This block corrects the output path for the final compiled library (.pyd).
# Without it, Visual Studio would create extra /Debug or /Release subfolders,
# breaking the setuptools build process.
set_target_properties(directxtex_decoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

# --- Linking ---
# Link our Python module against the DirectXTex static library.
target_link_libraries(directxtex_decoder PRIVATE DirectXTex)