param([string]$EnvName = "yolo-retraining-v5")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$YoloRoot = if ($env:YOLOV5_ROOT) { [IO.Path]::GetFullPath($env:YOLOV5_ROOT) } else { Join-Path $ProjectRoot ".third_party\yolov5" }
$escapedName = [regex]::Escape($EnvName)
if (-not (conda env list | Select-String -Pattern "^\s*$escapedName\s")) {
    throw "Conda environment '$EnvName' does not exist. Use scripts/bootstrap.ps1 explicitly."
}
$env:YOLOV5_ROOT = $YoloRoot
conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "Editable installation failed with exit code $LASTEXITCODE" }
conda run --no-capture-output -n $EnvName python -m yolo_retraining.doctor --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed with exit code $LASTEXITCODE" }
