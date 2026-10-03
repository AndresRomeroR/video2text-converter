$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$legacyPython = Join-Path $projectRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonExecutable = $venvPython
} elseif (Test-Path -LiteralPath $legacyPython -PathType Leaf) {
    $pythonExecutable = $legacyPython
} else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}

& $pythonExecutable (Join-Path $projectRoot "build_exe.py")
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo generar y verificar el ejecutable. Revisa el error anterior."
}
