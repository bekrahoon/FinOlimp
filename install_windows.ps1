$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = $PSScriptRoot
$McpProject = Join-Path $Root 'stage3_ai\integrations\finolimp_mcp'
$Requirements = Join-Path $Root 'requirements-runner.txt'

function Stop-Install([string]$Message) {
    Write-Error "ОШИБКА: $Message"
    exit 1
}

function Add-UserPathEntry([string]$Entry) {
    $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $Entries = @($UserPath -split ';' | Where-Object { $_ })
    $AlreadyPresent = $Entries | Where-Object { $_.TrimEnd('\') -ieq $Entry.TrimEnd('\') }
    if (-not $AlreadyPresent) {
        [Environment]::SetEnvironmentVariable('Path', (@($Entry) + $Entries) -join ';', 'User')
    }
    $CurrentEntries = @($env:Path -split ';' | Where-Object { $_ })
    $InCurrentPath = $CurrentEntries | Where-Object { $_.TrimEnd('\') -ieq $Entry.TrimEnd('\') }
    if (-not $InCurrentPath) { $env:Path = "$Entry;$env:Path" }
}

if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    Stop-Install 'Скрипт нужно запускать из полной копии проекта FinOlimp.'
}
if (-not (Test-Path -LiteralPath (Join-Path $McpProject 'pyproject.toml') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $McpProject 'uv.lock') -PathType Leaf)) {
    Stop-Install "Не найдены файлы MCP-проекта: $McpProject"
}

Write-Host "`n[1/5] Устанавливаю uv (менеджер Python и виртуальных окружений)..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
$UvCandidates = @(
    (Join-Path $HOME '.local\bin'),
    (Join-Path $HOME '.cargo\bin')
)
foreach ($BinDir in $UvCandidates) {
    if (Test-Path (Join-Path $BinDir 'uv.exe')) {
        Add-UserPathEntry $BinDir
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Stop-Install 'uv не найден после установки. Перезапустите PowerShell и повторите установку.'
}

Write-Host "`n[2/5] Устанавливаю Python 3.12 и зависимости Stage 1/2..."
uv python install 3.12
if ($LASTEXITCODE -ne 0) { Stop-Install 'Не удалось установить Python 3.12 через uv.' }
uv venv --python 3.12 (Join-Path $Root '.venv')
if ($LASTEXITCODE -ne 0) { Stop-Install 'Не удалось создать виртуальное окружение .venv.' }
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
uv pip install --python $VenvPython -r $Requirements
if ($LASTEXITCODE -ne 0) { Stop-Install 'Не удалось установить зависимости Stage 1/2.' }

Write-Host "`n[3/5] Создаю отдельное заблокированное окружение MCP..."
uv sync --locked --no-dev --project $McpProject --python 3.12
if ($LASTEXITCODE -ne 0) { Stop-Install 'Не удалось установить зависимости MCP.' }

Write-Host "`n[4/5] Устанавливаю OpenCode CLI для Windows..."
# Official standalone CLI archive documented at https://opencode.ai/v2/docs/.
$OpenCodeVersion = '2.0.6'
$Architecture = switch ($env:PROCESSOR_ARCHITECTURE) {
    'ARM64' { 'arm64' }
    'AMD64' { 'x64' }
    default { Stop-Install "Архитектура Windows '$env:PROCESSOR_ARCHITECTURE' не поддерживается этим установщиком." }
}
$InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\OpenCode'
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("finolimp-opencode-" + [guid]::NewGuid().ToString('N'))
$Archive = Join-Path $TempDir 'opencode.zip'
$ExtractDir = Join-Path $TempDir 'unpacked'
$DownloadUrl = "https://opencode.ai/files/bin/$OpenCodeVersion/opencode-windows-$Architecture.zip"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $Archive
    Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractDir -Force
    $Binary = Get-ChildItem -LiteralPath $ExtractDir -Filter 'opencode.exe' -File -Recurse | Select-Object -First 1
    if (-not $Binary) {
        $Binary = Get-ChildItem -LiteralPath $ExtractDir -Filter '*.exe' -File -Recurse |
            Where-Object { $_.BaseName -like 'opencode*' } | Select-Object -First 1
    }
    if (-not $Binary) { Stop-Install 'В официальном архиве OpenCode не найден исполняемый файл.' }
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -LiteralPath $Binary.FullName -Destination (Join-Path $InstallDir 'opencode.exe') -Force
}
finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force
    }
}

Add-UserPathEntry $InstallDir
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    Stop-Install 'OpenCode установлен, но не найден. Перезапустите PowerShell.'
}

Write-Host "`n[5/5] Проверяю установку..."
& opencode --version
if ($LASTEXITCODE -ne 0) { Stop-Install 'OpenCode установлен, но проверка версии завершилась ошибкой.' }

Write-Host "`nУстановка завершена. Откройте новый PowerShell, затем из корня проекта:"
Write-Host '  .\.venv\Scripts\python.exe .\run_stage_1_2.py'
Write-Host '  .\.venv\Scripts\python.exe .\run_ai.py'
Write-Host "`nПеред использованием AI подключите провайдера командой /connect в OpenCode."
