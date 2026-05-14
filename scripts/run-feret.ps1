param(
    [string]$Root = "experiments\feret",
    [string]$BuildDir = "build\vs2026-msvc",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$exe = Join-Path $BuildDir "$Configuration\face_cli.exe"
if (!(Test-Path $exe)) {
    throw "Executable not found: $exe"
}

$subsets = @("fa", "fc", "qr")
foreach ($subset in $subsets) {
    $gallery = Join-Path $Root "$subset\gallery"
    $probe = Join-Path $Root "$subset\probe"
    $output = Join-Path $Root "$subset\output"

    if (!(Test-Path $gallery)) { throw "Missing gallery folder: $gallery" }
    if (!(Test-Path $probe)) { throw "Missing probe folder: $probe" }

    $galleryFiles = Get-ChildItem -File $gallery -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|pgm)$' }
    $probeFiles = Get-ChildItem -File $probe -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|pgm)$' }
    if ($galleryFiles.Count -eq 0 -or $probeFiles.Count -eq 0) {
        Write-Host "Skipping ${subset}: gallery or probe is empty."
        continue
    }

    New-Item -ItemType Directory -Force -Path $output | Out-Null

    & $exe batch --gallery-dir $gallery --query-dir $probe --output-dir $output
}
