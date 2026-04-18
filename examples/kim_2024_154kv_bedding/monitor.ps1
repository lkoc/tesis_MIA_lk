# Monitor rapido — ejecutar en cualquier momento para ver estado de los 6 runs
# Uso: powershell .\examples\kim_2024_154kv_bedding\monitor.ps1

$base = Split-Path $MyInvocation.MyCommand.Path
$done = @{}

$runs = @(
    @{label="run_example     quick   "; log="$base\results\train.log";                    pt="$base\results\model_final.pt"},
    @{label="run_example     research"; log="$base\results_research\train.log";            pt="$base\results_research\model_final.pt"},
    @{label="run_pac         quick   "; log="$base\results_pac_quick\train.log";           pt="$base\results_pac_quick\model_final.pt"},
    @{label="run_pac         research"; log="$base\results_pac_research\train.log";        pt="$base\results_pac_research\model_final.pt"},
    @{label="run_multilayer  quick   "; log="$base\results_multilayer_quick\train.log";    pt="$base\results_multilayer_quick\model_case_B.pt"},
    @{label="run_multilayer  research"; log="$base\results_multilayer_research\train.log"; pt="$base\results_multilayer_research\model_case_B.pt"}
)

Write-Host "`n$(Get-Date -Format 'HH:mm:ss') — Estado de los 6 entrenamientos Kim 2024`n"
foreach ($r in $runs) {
    $status = if (Test-Path $r.pt) { "[DONE]" } else { "[run] " }
    if (Test-Path $r.log) {
        $last = (Get-Content $r.log | Select-Object -Last 1) -replace '^\d+:\d+:\d+ \[INFO\] ', ''
        Write-Host "  $status $($r.label) | $last"
    } else {
        Write-Host "  [wait] $($r.label) | (iniciando...)"
    }
}

$nPy = (Get-Process python -ErrorAction SilentlyContinue).Count
$totalCPU = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object CPU -Sum).Sum
Write-Host "`n  Procesos Python activos: $nPy  |  CPU total acumulado: $([math]::Round($totalCPU/60,1)) min"
