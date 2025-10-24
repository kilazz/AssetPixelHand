# setup.py
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension class that represents a C++ extension built with CMake.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        # Convert sourcedir to an absolute path
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


# A custom build command that runs CMake to build the extension.
class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # The path where setuptools expects to find the compiled file (.pyd)
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Get the path to pybind11's CMake files
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        # Arguments for the CMake configuration step
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # Use 'Release' for an optimized build
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
        ]

        # Arguments for the CMake build step
        build_args = ["--config", "Release"]

        # Temporary build directory
        build_temp = Path(self.build_temp) / ext.name
        if build_temp.exists():
            shutil.rmtree(build_temp)
        build_temp.mkdir(parents=True)

        print("--- Running CMake configuration ---")
        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        print("--- Building extension with CMake ---")
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)
        print("--- Build successful! ---")


# The main setup function
setup(
    name="directxtex_decoder",
    version="0.2.0",  # Incremented version for the fixes
    description="A Python wrapper for DirectXTex to decode various DDS files.",
    author="Zed",  # You can change this
    author_email="your.email@example.com",  # You can change this
    ext_modules=[CMakeExtension("directxtex_decoder", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
