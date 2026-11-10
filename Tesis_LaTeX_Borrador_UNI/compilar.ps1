$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    latexmk -lualatex -interaction=nonstopmode -file-line-error tesis.tex
}
finally {
    Pop-Location
}
