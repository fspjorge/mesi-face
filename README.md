# mesi-face

Implementation in C++ of the FACE method described in:

Maria De Marsico, Michele Nappi, Daniel Riccio, Harry Wechsler  
Robust Face Recognition for Uncontrolled Pose and Illumination Changes  
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2013.

This repository comes from my master's thesis work and is focused on reproducing the paper's pipeline as closely as possible on a modern Windows toolchain.

## Thesis Context

The goal of the original work was to study a face recognition pipeline robust to:

- pose changes
- illumination changes
- low quality samples
- uncertain recognition responses

The FACE method combines:

- facial landmark localization
- pose normalization
- illumination quality estimation
- illumination normalization with SQI
- local correlation matching
- response reliability measures

## Current State

The project now builds and runs on Windows with Visual Studio 2026 while keeping the paper's structure.

Main points:

- `face_cli` for command-line experiments
- `face_gui` for visual inspection
- official `STASM 4.1.0` integrated under `third_party/stasm`
- OpenCV 4 based build with `CMake` and `vcpkg`
- pose normalization, SQI generation, local correlation, `SP`, `SI`, `SRR1`, `SRR2`

## Fidelity to the Paper

The implementation is now explicitly aligned with the original thesis direction:

- landmark localization uses `STASM`
- the pipeline uses the converted 68-point STASM layout that the original code relied on
- pose normalization follows the historical project logic built around the paper
- SQI remains part of the matching pipeline and is not treated as a cosmetic image enhancement

Important note:

- the project is faithful in structure and implementation intent, but it is still an experimental reproduction, not a published benchmark reproduction
- results still depend heavily on gallery composition, image resolution, and landmark precision

## Repository Layout

```text
.
|-- data/                      # sample gallery images
|-- src/
|   |-- app_support.*         # shared CLI/GUI helpers
|   |-- face_pipeline.*       # STASM, SP, SI, pose, SQI
|   |-- face_matcher.*        # local/global correlation, SRR
|   |-- face_gui.cpp          # Win32 GUI
|   `-- main.cpp              # CLI
|-- third_party/stasm/        # vendored STASM sources, models, cascades, license
|-- CMakeLists.txt
|-- CMakePresets.json
|-- launch.vs.json
`-- vcpkg.json
```

## Build Requirements

- Windows
- Visual Studio 2026
- CMake
- Desktop C++ workload in Visual Studio

Dependencies are restored through `vcpkg`.

## Build

```powershell
cmake -S . -B build\vs2026-msvc -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/Program Files/Microsoft Visual Studio/18/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build\vs2026-msvc --config Release
```

Executables:

- `build\vs2026-msvc\Release\face_cli.exe`
- `build\vs2026-msvc\Release\face_gui.exe`

## CLI Usage

By default, STASM assets are loaded from:

- `third_party/stasm/data`

You can override that directory with `--cascade-dir` or `--model-dir`.

### Identify

```powershell
$env:PATH='C:\Work\mesi-face\build\vs2026-msvc\vcpkg_installed\x64-windows\bin;' + $env:PATH
.\build\vs2026-msvc\Release\face_cli.exe identify --gallery-dir data --query-image .\testface.jpg --output-dir output
```

### Train

```powershell
.\build\vs2026-msvc\Release\face_cli.exe train --gallery-dir data --output-dir output
```

### Batch

```powershell
.\build\vs2026-msvc\Release\face_cli.exe batch --gallery-dir data --query-dir data --output-dir output
```

## GUI

```powershell
$env:PATH='C:\Work\mesi-face\build\vs2026-msvc\vcpkg_installed\x64-windows\bin;' + $env:PATH
.\build\vs2026-msvc\Release\face_gui.exe
```

The GUI allows:

- selecting a gallery directory
- selecting a query image
- toggling illumination normalization
- visualizing original, pose-normalized, and SQI/matching views
- reading `SP`, `SI`, `SRR1`, `SRR2` and ranking results

## Generated Outputs

The output directory contains:

- `normalized/*_pose.png`
- `normalized/*_normalized.png`
- `normalized/*_sqi.png`
- `histograms/*`
- `batch_results.csv`

## Experimental Notes From the Thesis

The thesis experiments considered databases such as:

- `CDB`
- `LFW`
- `SCFace`
- `FERET`

Relevant observations from that work:

- performance depends strongly on the accuracy of facial landmark localization
- nose tip localization errors can seriously damage pose normalization
- low resolution images reduce the effectiveness of the full pipeline
- `SP`, `SI`, `SRR1`, and `SRR2` are useful as quality and confidence filters
- using very few images per identity is possible, but stability drops on uncontrolled data

## About Training Images

For paper-faithful runtime behavior, STASM itself does not need to be retrained for normal use in this repository because the bundled STASM model is already pretrained.

What does matter experimentally is:

- more gallery images per identity improve recognition robustness
- more varied gallery samples help when pose, makeup, lighting, and expression differ strongly
- if the goal is to reproduce thesis-level evaluation, a proper dataset split matters more than a single-image gallery

So if two photos of the same person still rank too close to other identities, the likely causes are:

- difficult illumination normalization
- strong appearance shift between samples
- too few gallery samples per identity
- limits of the original local-correlation approach on uncontrolled images

## License Notes

- the integrated STASM code keeps its original BSD-style license in `third_party/stasm/LICENSE.txt`
- OpenCV cascade files bundled inside the STASM data directory keep the original OpenCV licensing terms
