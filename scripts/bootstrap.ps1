param(
    [string]$EnvName = "yolo-cl",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu124",
    [switch]$AllowExistingEnvironmentUpdate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$escapedName = [regex]::Escape($EnvName)
$exists = [bool](conda env list | Select-String -Pattern "^\s*$escapedName\s")
if ($exists -and -not $AllowExistingEnvironmentUpdate) {
    throw "Environment '$EnvName' already exists. Use use_existing_env.ps1, or pass -AllowExistingEnvironmentUpdate explicitly."
}
if (-not $exists) {
    conda env create -n $EnvName -f (Join-Path $ProjectRoot "environment.yml")
}
conda run --no-capture-output -n $EnvName python -m pip install --index-url $TorchIndex torch==2.6.0 torchvision==0.21.0
conda run --no-capture-output -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements\runtime.lock")
conda run --no-capture-output -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements\dev.lock")
conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot --no-deps --no-build-isolation
conda run --no-capture-output -n $EnvName python -m yolo_retraining.doctor --project-root $ProjectRoot
