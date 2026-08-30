[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Source,
    [string]$Output = "dosbox-x-mcp.exe"
)

$ErrorActionPreference = "Stop"
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$tag = (& git -C $sourcePath describe --tags --exact-match 2>$null)
if ($LASTEXITCODE -ne 0 -or $tag -ne "dosbox-x-v2026.07.02") {
    throw "Source must be the exact DOSBox-X tag dosbox-x-v2026.07.02."
}

$include = Join-Path $PSScriptRoot "dosbox-x-mcp-v2.inc"
Copy-Item -LiteralPath $include -Destination (Join-Path $sourcePath "src\gui\dosbox-x-mcp-v2.inc") -Force
foreach ($patchName in @("dosbox-x-mcp-keyboard-v1.patch", "dosbox-x-mcp-write-watch.patch", "dosbox-x-mcp-irq-ack.patch", "dosbox-x-mcp-video-frame.patch", "dosbox-x-savestate-flag-loop.patch")) {
    $patch = Join-Path $PSScriptRoot $patchName
    & git -C $sourcePath apply --check $patch
    if ($LASTEXITCODE -eq 0) {
        & git -C $sourcePath apply $patch
    } else {
        & git -C $sourcePath apply --reverse --check $patch
        if ($LASTEXITCODE -ne 0) {
            throw "Companion patch $patchName neither applies cleanly nor is already applied."
        }
    }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$installation = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
if (-not $installation) {
    throw "Visual Studio Build Tools with MSBuild are required."
}
$msbuild = Join-Path $installation "MSBuild\Current\Bin\MSBuild.exe"
& $msbuild (Join-Path $sourcePath "vs\dosbox-x.sln") /m /t:Build "/p:Configuration=Release SDL2" /p:Platform=x64 /p:PlatformToolset=v143 /v:minimal
if ($LASTEXITCODE -ne 0) {
    throw "DOSBox-X build failed."
}

$built = Join-Path $sourcePath "bin\x64\Release SDL2\dosbox-x.exe"
Copy-Item -LiteralPath $built -Destination $Output -Force
Write-Output (Resolve-Path -LiteralPath $Output).Path
