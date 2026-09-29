$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Foundation, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
"Idiomas OCR disponibles: " + (([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object { $_.LanguageTag }) -join ', ')
$lang = [Windows.Globalization.Language]::new('es')
$engine = if ([Windows.Media.Ocr.OcrEngine]::IsLanguageSupported($lang)) { [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang) } else { [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
"Motor: " + $engine.RecognizerLanguage.LanguageTag
$dir = Join-Path $env:TEMP 'audit_gen\pages'
foreach ($tag in 'g01', 'g02') {
  $sb = New-Object System.Text.StringBuilder
  Get-ChildItem $dir -Filter "${tag}_p*.png" | Sort-Object Name | ForEach-Object {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($_.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $res = Await ($engine.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
    [void]$sb.AppendLine("`n=== $($_.BaseName) ===")
    foreach ($l in $res.Lines) { [void]$sb.AppendLine($l.Text) }
    $stream.Dispose()
  }
  [System.IO.File]::WriteAllText((Join-Path $env:TEMP "audit_gen\${tag}_ocr.txt"), $sb.ToString(), [System.Text.Encoding]::UTF8)
  "$tag OCR listo"
}
