[CmdletBinding()]
param(
    [string] $Python = "python",
    [switch] $ContinueOnError
)

$ErrorActionPreference = "Stop"

$HaraRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $HaraRoot "..\..")).Path

$TrackScripts = @(
    [pscustomobject]@{
        Name = "Deep SORT HBB"
        Path = Join-Path $HaraRoot "hara_deep_sort_HBB\demo.py"
    },
    [pscustomobject]@{
        Name = "Deep SORT OBB"
        Path = Join-Path $HaraRoot "hara_deep_sort_OBB\tracking_demo.py"
    },
    [pscustomobject]@{
        Name = "ByteTrack"
        Path = Join-Path $HaraRoot "hara_byte_track\demo.py"
    },
    [pscustomobject]@{
        Name = "BoT-SORT"
        Path = Join-Path $HaraRoot "hara_bot_sort\demo.py"
    }
)

foreach ($TrackScript in $TrackScripts) {
    if (-not (Test-Path -LiteralPath $TrackScript.Path -PathType Leaf)) {
        throw "Missing script for $($TrackScript.Name): $($TrackScript.Path)"
    }
}

$OriginalLocation = (Get-Location).Path
$OriginalPythonPath = $env:PYTHONPATH

try {
    $PythonPathEntries = @()
    if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $PythonPathEntries = $env:PYTHONPATH -split [System.IO.Path]::PathSeparator
    }

    if ($PythonPathEntries -notcontains $RepoRoot) {
        $env:PYTHONPATH = (@($RepoRoot) + $PythonPathEntries) -join [System.IO.Path]::PathSeparator
    }

    Set-Location -LiteralPath $HaraRoot

    foreach ($TrackScript in $TrackScripts) {
        $ScriptPath = (Resolve-Path -LiteralPath $TrackScript.Path).Path
        $StartedAt = Get-Date

        Write-Host ""
        Write-Host "[$($StartedAt.ToString('yyyy-MM-dd HH:mm:ss'))] Running $($TrackScript.Name)"
        Write-Host $ScriptPath

        & $Python $ScriptPath
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -ne 0) {
            $Message = "$($TrackScript.Name) failed with exit code $ExitCode."
            if ($ContinueOnError) {
                Write-Warning $Message
                continue
            }

            throw $Message
        }

        $Elapsed = (Get-Date) - $StartedAt
        Write-Host "Completed $($TrackScript.Name) in $($Elapsed.ToString('hh\:mm\:ss'))."
    }
}
finally {
    Set-Location -LiteralPath $OriginalLocation
    $env:PYTHONPATH = $OriginalPythonPath
}
