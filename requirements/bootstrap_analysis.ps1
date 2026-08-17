param(
    [string]$SourceEnvName = "yolo-retraining-v5",
    [string]$EnvName = "yolo-result-analysis",
    [switch]$AllowExistingEnvironmentUpdate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Test-CondaEnvironment([string]$Name) {
    $escapedName = [regex]::Escape($Name)
    return [bool](conda env list | Select-String -Pattern "^\s*$escapedName\s")
}

if (-not (Test-CondaEnvironment $SourceEnvName)) {
    throw "Source environment '$SourceEnvName' does not exist. Restore the training environment first."
}

$exists = Test-CondaEnvironment $EnvName
if ($exists -and -not $AllowExistingEnvironmentUpdate) {
    throw "Environment '$EnvName' already exists. Pass -AllowExistingEnvironmentUpdate to update it."
}

if (-not $exists) {
    conda create -n $EnvName --clone $SourceEnvName -y
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone '$SourceEnvName' into '$EnvName'" }
}

conda run --no-capture-output -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements\analysis.lock")
if ($LASTEXITCODE -ne 0) { throw "Analysis dependency installation failed" }

conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "Editable installation failed" }

conda run --no-capture-output -n $EnvName python -c "import cv2,numpy,onnx,onnxruntime,torch,torchvision; x=numpy.zeros((2,2),dtype=numpy.float32); assert torch.from_numpy(x).shape==(2,2); print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('cuda',torch.version.cuda); print('numpy',numpy.__version__); print('opencv',cv2.__version__); print('onnx',onnx.__version__); print('onnxruntime',onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Analysis environment validation failed" }
