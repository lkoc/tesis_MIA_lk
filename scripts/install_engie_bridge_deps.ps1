# Instala Playwright en el venv dedicado del bridge de engie.ai.
# Se ejecuta aparte para que la descarga larga (38 MB) sobreviva al timeout del
# comando que la invoca. Verifica al final que el modulo sea importable.
$ErrorActionPreference = 'Continue'
$Venv = 'c:\usr\ths_mia_fiis\tesis_MIA_lk\.venv-engie'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Log = 'c:\usr\ths_mia_fiis\tesis_MIA_lk\_pip_install.log'

"[$(Get-Date -Format o)] inicio de instalacion" | Out-File -Encoding utf8 $Log
& $Py -m pip install --no-input --disable-pip-version-check playwright *>&1 |
    Out-File -Encoding utf8 -Append $Log
"[$(Get-Date -Format o)] pip exit=$LASTEXITCODE" | Out-File -Encoding utf8 -Append $Log

& $Py -c "import playwright; print('PLAYWRIGHT_IMPORT_OK')" *>&1 |
    Out-File -Encoding utf8 -Append $Log
"[$(pi=$LASTEXITCODE; Get-Date -Format o)] import exit=$pi" |
    Out-File -Encoding utf8 -Append $Log
