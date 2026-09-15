# Remove only identified obsolete build products; archive unique old plan copies.
# Native PowerShell throughout; never follow a path outside this workspace.
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskAudit = Join-Path $taskRoot 'docs/auditoria/metodologia_2026-09-14'
function Assert-WorkspacePath([string]$candidate) {
    $absolute = [IO.Path]::GetFullPath($candidate)
    if (-not $absolute.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing operation outside workspace: $absolute"
    }
    if ((Get-Item -LiteralPath $absolute).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing reparse point' }
    return $absolute
}
$taskPlan = Join-Path $taskRoot 'Plan'
$unique = @('plan_tesis_cables_pinn copy.pdf','plan_tesis_cables_pinn.tex.bkp','plan_tesis_cables_pinn_backup.tex.bkp') |
    ForEach-Object { Join-Path $taskPlan $_ } | Where-Object { Test-Path -LiteralPath $_ }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archivePath = Join-Path $taskRoot 'docs/historico/plan_respaldos_previos_2026-09-14.zip'
$removed = @()
if ($unique.Count -gt 0) {
    if (Test-Path -LiteralPath $archivePath) { throw 'Archive already exists; inspect it before changing preserved evidence' }
    $archive = [IO.Compression.ZipFile]::Open($archivePath, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($candidate in $unique) {
            $absolute = Assert-WorkspacePath $candidate
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$absolute,[IO.Path]::GetFileName($absolute)) | Out-Null
        }
    } finally { $archive.Dispose() }
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($candidate in $unique) {
            $entry = $archive.GetEntry([IO.Path]::GetFileName($candidate))
            $stream = $entry.Open(); $algorithm = [Security.Cryptography.SHA256]::Create()
            try { $digest = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','') }
            finally { $stream.Dispose(); $algorithm.Dispose() }
            if ($digest -ne (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash) { throw 'Archive verification failed' }
        }
    } finally { $archive.Dispose() }
}
$generated = Get-ChildItem -LiteralPath $taskPlan -File | Where-Object {
    ($_.Name -like 'plan_tesis_cables_pinn copy.*' -and $_.Extension -ne '.pdf') -or
    $_.Name -like 'plan_tesis_verify_*' -or $_.Name -like '*SAVE-ERROR' -or $_.Name -eq '_verify.txt'
}
foreach ($candidate in @($unique) + @($generated.FullName)) {
    if (-not $candidate) { continue }
    $absolute = Assert-WorkspacePath $candidate
    $item = Get-Item -LiteralPath $absolute
    $removed += [PSCustomObject]@{path=$absolute.Substring($taskRoot.Length+1);bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $absolute).Hash;reason='verified archive or obsolete LaTeX auxiliary'}
}
$obsolete = Join-Path $taskRoot 'Benchmarks/explicit_results/coaxial_full/fixed/reference'
if (Test-Path -LiteralPath $obsolete) {
    $absolute = Assert-WorkspacePath $obsolete
    $corrected = Join-Path $taskRoot 'Benchmarks/explicit_results/coaxial_full/fixed/reference_verified/fem_l2.json'
    if (-not (Test-Path -LiteralPath $corrected)) { throw 'Corrected reference is missing' }
    $proof = Get-Content -LiteralPath $corrected -Raw | ConvertFrom-Json
    if ($proof.error_Tmax_exact_K -gt 0.00001) { throw 'Corrected reference failed analytic verification' }
    foreach ($item in Get-ChildItem -LiteralPath $absolute -File -Recurse) {
        $checked = Assert-WorkspacePath $item.FullName
        $removed += [PSCustomObject]@{path=$checked.Substring($taskRoot.Length+1);bytes=$item.Length;sha256=(Get-FileHash -LiteralPath $checked).Hash;reason='superseded pilot with incorrect material interpolation; corrected reference_verified retained'}
        if ($item.Extension -eq '.json' -and $item.Name -like 'fem_l*') {
            Copy-Item -LiteralPath $checked -Destination (Join-Path $taskAudit ('invalid_interpolation_'+$item.Name))
        }
    }
}
if ($removed.Count -eq 0) { Write-Output 'No identified obsolete files remain'; exit }
$manifest = [PSCustomObject]@{utc=[DateTime]::UtcNow.ToString('o');archive=$archivePath;files=$removed;total_bytes=($removed | Measure-Object bytes -Sum).Sum}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskAudit 'cleanup.json') -Encoding UTF8
foreach ($entry in $removed) {
    $absolute = Assert-WorkspacePath (Join-Path $taskRoot $entry.path)
    Remove-Item -LiteralPath $absolute
}
if (Test-Path -LiteralPath $obsolete) {
    # All contents are known and verified above; final absolute target is checked.
    $absolute = Assert-WorkspacePath $obsolete
    Remove-Item -LiteralPath $absolute -Recurse
}
Write-Output ("Cleaned {0} files, {1} bytes; unique backups verified in archive" -f $removed.Count,$manifest.total_bytes)
