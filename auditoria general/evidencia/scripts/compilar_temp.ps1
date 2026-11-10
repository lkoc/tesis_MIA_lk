# Compila la tesis o el Plan en una copia temporal, sin tocar .aux/.pdf del repositorio.
# Uso: powershell -File compilar_temp.ps1 -Que tesis|plan [-Extra "\setstretch{1.4}"]
param(
  [ValidateSet('tesis','plan')][string]$Que = 'tesis',
  [string]$Salida = "$env:TEMP\ths_build"
)
$raiz = 'C:\usr\ths_mia_fiis\tesis_MIA_lk'
if ($Que -eq 'tesis') { $carpeta = 'Tesis_LaTeX_Borrador_UNI'; $principal = 'tesis.tex' } else { $carpeta = 'Plan'; $principal = 'plan_tesis_cables_pinn.tex' }
$destino = Join-Path $Salida $carpeta
if (Test-Path $destino) { Remove-Item $destino -Recurse -Force }
New-Item -ItemType Directory -Force -Path $destino | Out-Null
$pdfPrincipal = $principal -replace '\.tex$', '.pdf'
$excluirDirs = @('.git', 'envío_rev0', 'envío_rev1', 'envio_rev0', 'old', 'guias', 'literatura', 'datos_fuente', 'presentación', 'ejemplos_plan', '_audit_pages', '_fuentes_tmp', '_tmp_extracts', '_verify_pages', 'Tarea 1', 'Tarea 2', 'build', 'check')
robocopy (Join-Path $raiz $carpeta) $destino /E /XD $excluirDirs /XF *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.loe *.lop *.bcf *.run.xml *.out *.fls *.fdb_latexmk *.xdv $pdfPrincipal /NFL /NDL /NJH /NJS | Out-Null
# los PDF de la raiz son bibliografia y entregas; las figuras estan en subcarpetas
Get-ChildItem $destino -Filter *.pdf -File | Remove-Item -Force
$bibSel = Join-Path $raiz 'Benchmarks\references_selection.bib'
if (Test-Path $bibSel) {
  New-Item -ItemType Directory -Force -Path (Join-Path $Salida 'Benchmarks') | Out-Null
  Copy-Item $bibSel (Join-Path $Salida 'Benchmarks\references_selection.bib') -Force
}
Push-Location $destino
latexmk -lualatex -interaction=nonstopmode -halt-on-error $principal 2>&1 | Out-File -Encoding utf8 (Join-Path $Salida "$Que.compilacion.txt")
Pop-Location
Write-Output "Compilado en $destino"
