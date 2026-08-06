param([string]$EnvName = "yolo-cl")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$escapedName = [regex]::Escape($EnvName)
if (-not (conda env list | Select-String -Pattern "^\s*$escapedName\s")) {
    throw "Conda environment '$EnvName' does not exist. Use scripts/bootstrap.ps1 explicitly."
}

conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "Editable installation failed with exit code $LASTEXITCODE" }
conda run --no-capture-output -n $EnvName python -m yolo_retraining.doctor --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed with exit code $LASTEXITCODE" }
