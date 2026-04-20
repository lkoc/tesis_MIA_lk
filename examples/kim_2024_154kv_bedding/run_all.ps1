# Lanza los 6 entrenamientos Kim 2024 como procesos en segundo plano.
# Uso: powershell .\examples\kim_2024_154kv_bedding\run_all.ps1
# Monitorear avance: powershell .\examples\kim_2024_154kv_bedding\monitor.ps1

$base = Split-Path $MyInvocation.MyCommand.Path
$root = (Resolve-Path "$base\..\..\").Path
$python = (Get-Command python -ErrorAction Stop).Source

$runs = @(
    @{script="run_example.py";      profile="quick"},
    @{script="run_example.py";      profile="research"},
    @{script="run_research_pac.py"; profile="quick"},
    @{script="run_research_pac.py"; profile="research"},
    @{script="run_multilayer.py";   profile="quick"},
    @{script="run_multilayer.py";   profile="research"}
)

Write-Host "`nLanzando 6 entrenamientos desde $root`n"
foreach ($r in $runs) {
    $scriptPath = Join-Path $base $r.script
    $label = "$($r.script) --profile $($r.profile)"
    Write-Host "  -> $label"
    Start-Process $python `
        -ArgumentList "`"$scriptPath`" --profile $($r.profile)" `
        -WorkingDirectory $root `
        -WindowStyle Hidden
}

Write-Host "`nTodos los procesos iniciados."
Write-Host "Monitorear: powershell .\examples\kim_2024_154kv_bedding\monitor.ps1`n"
