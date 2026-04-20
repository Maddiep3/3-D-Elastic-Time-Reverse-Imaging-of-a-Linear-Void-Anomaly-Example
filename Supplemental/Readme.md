# Supplemental Files

C and CUDA source files used by the SConstruct workflows. Copyright and authorship information is included in the preamble of each file where provided.

## Prerequisites

- GCC
- NVIDIA CUDA Toolkit (a CUDA-capable GPU is required for `.cu` files)

## Building Executables

To compile the source files, run the following in your terminal:

- **C files:** `gcc myfile.c -o myfile.x -lm`
- **CUDA files:** `nvcc myfile.cu -o myfile.x -lm`

Replace `myfile` with the actual filename.

## Setup

These executables must be compiled **before** running any SConstruct workflows. After building, update the file paths in the relevant SConstruct files to point to your compiled executables.