Add-Type -AssemblyName System.Windows.Forms

$folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
$folderBrowser.Description = "Select the folder with files to rename"
$folderBrowser.ShowDialog() | Out-Null
$folderPath = $folderBrowser.SelectedPath

if (-Not $folderPath) {
    Write-Host "No folder selected. Exiting."
    exit
}

Set-Location $folderPath

Get-ChildItem -File | Where-Object { $_.Name -match ' ' } | ForEach-Object {
    $newName = $_.Name -replace ' ', '_'
    Rename-Item $_.FullName $newName
}

Write-Host "Done replacing spaces with underscores in: $folderPath"
