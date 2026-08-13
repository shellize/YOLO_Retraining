param(
    [string]$EnvName = "yolo-retraining-v5",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu121",
    [switch]$AllowExistingEnvironmentUpdate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$YoloCommit = "915bbf294bb74c859f0b41f1c23bc395014ea679"
$YoloRoot = if ($env:YOLOV5_ROOT) { [IO.Path]::GetFullPath($env:YOLOV5_ROOT) } else { Join-Path $ProjectRoot ".third_party\yolov5" }
$WheelRoot = Join-Path $ProjectRoot ".third_party\wheels"
$escapedName = [regex]::Escape($EnvName)
$exists = [bool](conda env list | Select-String -Pattern "^\s*$escapedName\s")

if ($exists -and -not $AllowExistingEnvironmentUpdate) {
    throw "Environment '$EnvName' already exists. Pass -AllowExistingEnvironmentUpdate explicitly to verify/update project packages."
}
if (-not $exists) {
    conda env create -n $EnvName -f (Join-Path $ProjectRoot "environment.yml")
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed" }
}

if (-not (Test-Path -LiteralPath $YoloRoot)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $YoloRoot) | Out-Null
    git clone https://github.com/ultralytics/yolov5.git $YoloRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone YOLOv5" }
    git -C $YoloRoot checkout --detach $YoloCommit
    if ($LASTEXITCODE -ne 0) { throw "Failed to checkout YOLOv5 commit $YoloCommit" }
}

$actualCommit = (git -C $YoloRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $YoloCommit) {
    throw "YOLOv5 revision mismatch at '$YoloRoot': expected $YoloCommit, found $actualCommit"
}
$trackedChanges = git -C $YoloRoot status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $trackedChanges) {
    throw "YOLOv5 source at '$YoloRoot' contains tracked modifications"
}

$Weights = Join-Path $YoloRoot "yolov5s.pt"
if (-not (Test-Path -LiteralPath $Weights)) {
    Invoke-WebRequest -Uri "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt" -OutFile $Weights
}
if ((Get-Item -LiteralPath $Weights).Length -le 0) { throw "Downloaded YOLOv5s checkpoint is empty: $Weights" }

New-Item -ItemType Directory -Force -Path $WheelRoot | Out-Null
$TorchWheel = Join-Path $WheelRoot "torch-2.2.2+cu121-cp310-cp310-win_amd64.whl"
$TorchvisionWheel = Join-Path $WheelRoot "torchvision-0.17.2+cu121-cp310-cp310-win_amd64.whl"
$TorchUrl = if ($env:YOLO_RETRAINING_TORCH_WHEEL_URL) { $env:YOLO_RETRAINING_TORCH_WHEEL_URL } else { "$TorchIndex/torch-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl" }
$TorchvisionUrl = if ($env:YOLO_RETRAINING_TORCHVISION_WHEEL_URL) { $env:YOLO_RETRAINING_TORCHVISION_WHEEL_URL } else { "$TorchIndex/torchvision-0.17.2%2Bcu121-cp310-cp310-win_amd64.whl" }
$Downloads = @(
    @{ Url = $TorchUrl; Path = $TorchWheel },
    @{ Url = $TorchvisionUrl; Path = $TorchvisionWheel }
)
foreach ($Download in $Downloads) {
    conda run -n $EnvName python -m zipfile -t $Download.Path *> $null
    if ($LASTEXITCODE -ne 0) {
        curl.exe --fail --location --retry 20 --retry-all-errors --continue-at - --output $Download.Path $Download.Url
        if ($LASTEXITCODE -ne 0) { throw "Wheel download failed: $($Download.Url)" }
        conda run -n $EnvName python -m zipfile -t $Download.Path *> $null
        if ($LASTEXITCODE -ne 0) { throw "Downloaded wheel is incomplete or invalid: $($Download.Path)" }
    }
}
conda run --no-capture-output -n $EnvName python -m pip install $TorchWheel $TorchvisionWheel
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
conda run --no-capture-output -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements\runtime.lock")
if ($LASTEXITCODE -ne 0) { throw "Runtime dependency installation failed" }
conda run --no-capture-output -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements\dev.lock")
if ($LASTEXITCODE -ne 0) { throw "Development dependency installation failed" }
conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "Editable installation failed" }

$env:YOLOV5_ROOT = $YoloRoot
conda run --no-capture-output -n $EnvName python -m yolo_retraining.doctor --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Environment validation failed" }
