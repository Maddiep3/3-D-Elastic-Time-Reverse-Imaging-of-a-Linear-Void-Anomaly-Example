# 3D Elastic Time-Reverse Imaging of a Linear Void Anomaly — Example

2D synthetic examples from the paper "3-D Elastic Time-Reverse Imaging of a Linear Void Anomaly."

> Citation forthcoming. DOI will be added upon publication.

## Overview

This repository contains reproducible SConstruct workflows for 2D synthetic elastic time-reverse imaging (E-TRI) examples. E-TRI is a migration technique that uses multicomponent elastic wavefields to image subsurface anomalies by back-propagating recorded data through a modeled medium.

## Prerequisites

- [Madagascar](https://ahay.org/wiki/Main_Page) (open-source seismic processing and reproducible research framework)
- GCC (for compiling `.c` files)
- NVIDIA CUDA Toolkit (for compiling `.cu` files — requires a CUDA-capable GPU)

## Repository Structure

- **SConstruct/** — Zipped SConstruct workflows for each synthetic example
- **Supplemental/** — C and CUDA source files required to build models and run E-TRI migrations

## Getting Started

1. Install the prerequisites listed above.
2. Build the supplemental executables (see `Supplemental/README.md`).
3. Update file paths in the SConstructs to point to your compiled executables.
4. Run the examples (see `SConstruct/README.md`).

## License

Third-party source files retain their original copyright and authorship as noted in each file's preamble.