param(
    [string]$SourceRoot = "",
    [string]$TargetRoot = "experiments\feret",
    [string]$Manifest = "experiments\feret\manifest.csv",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Ensure-SubsetLayout {
    param([string]$Root)

    foreach ($subset in @("fa", "fc", "qr")) {
        foreach ($leaf in @("gallery", "probe", "output")) {
            $path = Join-Path $Root "$subset\$leaf"
            New-Item -ItemType Directory -Force -Path $path | Out-Null
        }
    }
}

function Copy-ManifestEntry {
    param(
        [pscustomobject]$Row,
        [string]$SourceRoot,
        [string]$TargetRoot,
        [switch]$Force
    )

    $subset = $Row.Subset.Trim().ToLowerInvariant()
    $role = $Row.Role.Trim().ToLowerInvariant()
    $sourceValue = $Row.Source.Trim()
    $targetName = if ($Row.TargetName) { $Row.TargetName.Trim() } else { Split-Path $sourceValue -Leaf }

    if ($subset -notin @("fa", "fc", "qr")) {
        throw "Invalid subset in manifest: $subset"
    }
    if ($role -notin @("gallery", "probe")) {
        throw "Invalid role in manifest: $role"
    }

    $destinationDir = Join-Path $TargetRoot "$subset\$role"
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    $sourcePath = if ([System.IO.Path]::IsPathRooted($sourceValue)) {
        $sourceValue
    } else {
        Join-Path $SourceRoot $sourceValue
    }

    if (!(Test-Path $sourcePath)) {
        throw "Source file not found: $sourcePath"
    }

    $destinationPath = Join-Path $destinationDir $targetName
    if ((Test-Path $destinationPath) -and -not $Force) {
        throw "Destination exists: $destinationPath. Use -Force to overwrite."
    }

    Copy-Item -Force:$Force -LiteralPath $sourcePath -Destination $destinationPath
}

if (!(Test-Path $Manifest)) {
    $manifestDir = Split-Path $Manifest -Parent
    if ($manifestDir) {
        New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
    }
    $template = @"
Subset,Role,Source,TargetName
fa,gallery,fa\001_frontal_1.jpg,001_frontal_1.jpg
fa,probe,fa\001_expression_1.jpg,001_expression_1.jpg
fc,gallery,fc\001_frontal_1.jpg,001_frontal_1.jpg
fc,probe,fc\001_illumination_1.jpg,001_illumination_1.jpg
qr,gallery,qr\001_frontal_1.jpg,001_frontal_1.jpg
qr,probe,qr\001_pose_right_1.jpg,001_pose_right_1.jpg
"@
    Set-Content -LiteralPath $Manifest -Value $template
    Write-Host "Created manifest template at $Manifest"
    Write-Host "Fill it with your FERET file mapping and rerun this script."
    exit 0
}

Ensure-SubsetLayout -Root $TargetRoot

$rows = Import-Csv $Manifest
if ($rows.Count -eq 0) {
    throw "Manifest is empty: $Manifest"
}

foreach ($row in $rows) {
    Copy-ManifestEntry -Row $row -SourceRoot $SourceRoot -TargetRoot $TargetRoot -Force:$Force
}

Write-Host "FERET structure prepared under $TargetRoot"
