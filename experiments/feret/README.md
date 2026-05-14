# FERET Experimental Protocol

This folder contains a working structure for the thesis experiments based on the FERET subsets referenced in the FACE paper.

## Subsets

- `fa` for frontal faces with expression variation
- `fc` for frontal faces with illumination variation
- `qr` for pose variation toward the right

## Folder Layout

Each subset uses the same structure:

```text
experiments/feret/
  fa/
    gallery/
    probe/
    output/
  fc/
    gallery/
    probe/
    output/
  qr/
    gallery/
    probe/
    output/
```

## How To Use

First run [scripts/prepare-feret.ps1](../../scripts/prepare-feret.ps1) to generate `manifest.csv` if it does not exist yet. Fill the manifest with the source files you want copied into each subset.

Then run the same script again to populate the `gallery` and `probe` folders.

After that, run the CLI batch mode for each subset.

Example:

```powershell
.\scripts\prepare-feret.ps1
.\build\vs2026-msvc\Release\face_cli.exe batch --gallery-dir experiments\feret\fa\gallery --query-dir experiments\feret\fa\probe --output-dir experiments\feret\fa\output
.\build\vs2026-msvc\Release\face_cli.exe batch --gallery-dir experiments\feret\fc\gallery --query-dir experiments\feret\fc\probe --output-dir experiments\feret\fc\output
.\build\vs2026-msvc\Release\face_cli.exe batch --gallery-dir experiments\feret\qr\gallery --query-dir experiments\feret\qr\probe --output-dir experiments\feret\qr\output
```

## Notes

- Keep the gallery/probe split fixed while comparing results.
- Use the same `--cascade-dir` or leave it pointed at the bundled `third_party/stasm/data`.
- Store the exported CSV results in the corresponding `output` folder.
