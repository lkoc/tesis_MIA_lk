# Remove superseded thesis inputs only after archive and active-input checks.
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskThesis = Join-Path $taskRoot 'Tesis_LaTeX_Borrador_UNI'
$taskArchive = Join-Path $taskRoot 'docs/historico/tesis_antes_fisica_explicita_2026-09-15.zip'
$taskAudit = Join-Path $taskRoot 'docs/auditoria/revision_integral_2026-09-15'
$candidates = @(Get-ChildItem -LiteralPath (Join-Path $taskThesis 'tablas') -Filter 'benchmark_*.tex' -File)
foreach ($name in @('03_analisis_numerico_metodo.tex','03_muestreo_adaptativo.tex')) {
    $path = Join-Path $taskThesis ('capitulos/' + $name)
    if (Test-Path -LiteralPath $path) { $candidates += Get-Item -LiteralPath $path }
}
if ($candidates.Count -eq 0) { Write-Output 'No superseded thesis inputs remain'; exit }
$candidatePaths = @($candidates.FullName)
$activeText = (Get-ChildItem -LiteralPath $taskThesis -Filter '*.tex' -File -Recurse |
    Where-Object { $candidatePaths -notcontains $_.FullName } |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }) -join "`n"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($taskArchive)
$manifest = @()
try {
    foreach ($item in $candidates) {
        $absolute = [IO.Path]::GetFullPath($item.FullName)
        if (-not $absolute.StartsWith($taskThesis + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside thesis directory' }
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing reparse point' }
        if ($activeText -match [Regex]::Escape($item.BaseName)) { throw "An active source still names $($item.Name)" }
        $relative = $absolute.Substring($taskRoot.Length+1).Replace('\','/')
        $entry = $archive.GetEntry($relative)
        if ($null -eq $entry) { throw "Missing archive entry: $relative" }
        $stream = $entry.Open(); $algorithm = [Security.Cryptography.SHA256]::Create()
        try { $digest = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','') }
        finally { $stream.Dispose(); $algorithm.Dispose() }
        if ($digest -ne (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash) { throw "Archive differs: $relative" }
        $manifest += [PSCustomObject]@{path=$relative;bytes=$item.Length;sha256=$digest;reason='superseded reduced-model thesis input; verified archive retained'}
    }
} finally { $archive.Dispose() }
New-Item -ItemType Directory -Path $taskAudit -Force | Out-Null
[PSCustomObject]@{utc=[DateTime]::UtcNow.ToString('o');archive=$taskArchive;files=$manifest} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskAudit 'cleanup.json') -Encoding UTF8
foreach ($item in $candidates) { Remove-Item -LiteralPath $item.FullName }
Write-Output ("Removed {0} superseded inputs after SHA-256 archive verification" -f $manifest.Count)
