param(
    [ValidateSet("directml", "cuda", "cpu")]
    [string]$Profile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$profileLabels = @{
    "directml" = "GPU (DirectML)"
    "cuda" = "GPU (CUDA)"
    "cpu" = "CPU"
}

$requirementsByProfile = @{
    "directml" = "requirements.gpu.directml.txt"
    "cuda" = "requirements.gpu.cuda.txt"
    "cpu" = "requirements.cpu.txt"
}

function Test-IsWindowsStoreAlias {
    param(
        [string]$CommandPath
    )

    if (-not $CommandPath) {
        return $false
    }

    $normalized = $CommandPath.ToLowerInvariant()
    return $normalized -like "*\windowsapps\python*.exe" -or $normalized -like "*\windowsapps\py.exe"
}

function Get-SystemPythonInvoker {
    $invokers = @()

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand -and -not (Test-IsWindowsStoreAlias $pyCommand.Source)) {
        $invokers += [PSCustomObject]@{
            Name = "py -3"
            ExePath = $pyCommand.Source
            PrefixArgs = @("-3")
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and -not (Test-IsWindowsStoreAlias $pythonCommand.Source)) {
        $invokers += [PSCustomObject]@{
            Name = $pythonCommand.Source
            ExePath = $pythonCommand.Source
            PrefixArgs = @()
        }
    }

    $searchRoots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        (Join-Path $env:ProgramFiles "Python"),
        (Join-Path ${env:ProgramFiles(x86)} "Python")
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($root in $searchRoots) {
        $candidates = Get-ChildItem -Path $root -Filter python.exe -File -Recurse -ErrorAction SilentlyContinue
        foreach ($candidate in $candidates) {
            if (Test-IsWindowsStoreAlias $candidate.FullName) {
                continue
            }

            $alreadyAdded = $invokers | Where-Object { $_.ExePath -ieq $candidate.FullName } | Select-Object -First 1
            if ($alreadyAdded) {
                continue
            }

            $invokers += [PSCustomObject]@{
                Name = $candidate.FullName
                ExePath = $candidate.FullName
                PrefixArgs = @()
            }
        }
    }

    foreach ($invoker in $invokers) {
        & $invoker.ExePath @($invoker.PrefixArgs + @("-c", "import sys")) *> $null
        if ($LASTEXITCODE -eq 0) {
            return $invoker
        }
    }

    return $null
}

function Invoke-SystemPython {
    param(
        [string[]]$Args
    )

    $pythonInvoker = Get-SystemPythonInvoker
    if (-not $pythonInvoker) {
        throw "Could not find a usable Python interpreter. If Windows opens the Microsoft Store, disable App execution aliases for python.exe/python3.exe or install Python from python.org."
    }

    & $pythonInvoker.ExePath @($pythonInvoker.PrefixArgs + $Args)
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed via '$($pythonInvoker.Name)': $([string]::Join(' ', $Args))"
    }
}

if (-not $Profile) {
    Write-Host ""
    Write-Host "Select dependency profile:"
    Write-Host "1) GPU (DirectML, Recommended)"
    Write-Host "2) GPU (CUDA)"
    Write-Host "3) CPU"

    while (-not $Profile) {
        $choice = (Read-Host "Enter 1-3").Trim()
        switch ($choice) {
            "1" { $Profile = "directml" }
            "2" { $Profile = "cuda" }
            "3" { $Profile = "cpu" }
            default { Write-Host "Invalid choice. Enter 1, 2, or 3." }
        }
    }
}

$requirementsFile = $requirementsByProfile[$Profile]
$requirementsPath = Join-Path $scriptDir $requirementsFile
if (-not (Test-Path $requirementsPath)) {
    throw "Missing requirements file: $requirementsPath"
}

$venvPath = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host ""
    Write-Host "Creating virtual environment at .venv ..."
    Invoke-SystemPython @("-m", "venv", $venvPath)
}

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed. Missing interpreter: $venvPython"
}

Write-Host ""
Write-Host "Ensuring pip is available in .venv ..."
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "Failed to bootstrap pip in .venv."
}

Write-Host ""
Write-Host "Upgrading packaging tools ..."
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools/wheel."
}

Write-Host ""
Write-Host "Installing $($profileLabels[$Profile]) dependencies ..."
& $venvPython -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed for profile '$Profile'."
}

Write-Host ""
Write-Host "Install complete."
Write-Host "Run the app with:"
Write-Host "  .\.venv\Scripts\python.exe main.py"
