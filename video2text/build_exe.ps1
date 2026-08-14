$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$legacyPython = Join-Path $projectRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonExecutable = $venvPython
} elseif (Test-Path -LiteralPath $legacyPython -PathType Leaf) {
    $pythonExecutable = $legacyPython
} else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}

$iconPath = Join-Path $projectRoot "totext.ico"
if (-not (Test-Path -LiteralPath $iconPath -PathType Leaf)) {
    throw "No se encontró el icono requerido: $iconPath"
}

$ffmpegPath = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobePath = (Get-Command ffprobe -ErrorAction Stop).Source

function Remove-ProjectArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)

    $absolutePath = [System.IO.Path]::GetFullPath($Path)
    $projectPrefix = $projectRoot.TrimEnd('\') + '\'
    if (-not $absolutePath.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Se rechazó una ruta fuera del proyecto: $absolutePath"
    }
    if (Test-Path -LiteralPath $absolutePath) {
        Remove-Item -LiteralPath $absolutePath -Recurse -Force
    }
}

& $pythonExecutable -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo instalar PyInstaller."
}

Remove-ProjectArtifact (Join-Path $projectRoot "build")
Remove-ProjectArtifact (Join-Path $projectRoot "dist")
Remove-ProjectArtifact (Join-Path $projectRoot "Video2Text.spec")

Get-ChildItem -LiteralPath $projectRoot -Recurse -Directory -Filter "__pycache__" |
    ForEach-Object { Remove-ProjectArtifact $_.FullName }

& $pythonExecutable -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "Video2Text" `
    --icon $iconPath `
    --add-data "$iconPath;." `
    --add-binary "$ffmpegPath;." `
    --add-binary "$ffprobePath;." `
    --hidden-import tkinterdnd2 `
    --collect-all tkinterdnd2 `
    --collect-all whisper `
    (Join-Path $projectRoot "video2text.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller no pudo generar el ejecutable."
}

$executablePath = Join-Path $projectRoot "dist\Video2Text.exe"
if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
    throw "La compilación terminó sin generar: $executablePath"
}

Write-Output "Ejecutable generado: $executablePath"
