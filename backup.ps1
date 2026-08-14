<#
  stock_history.db backup script
  Usage: powershell -ExecutionPolicy Bypass -File backup.ps1
  1. Compress data/stock_history.db -> backups/stock_history_YYYYMMDD.tar.gz
  2. Auto-upload to GitHub Releases if gh CLI is installed & authenticated
  3. Otherwise print manual upload instructions
#>
param(
    [string]$Repo = "HonorQ/StockPrediction"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DbPath = Join-Path $Root "data\stock_history.db"
$BackupDir = Join-Path $Root "backups"

if (-not (Test-Path $DbPath)) {
    Write-Host "[ERROR] data/stock_history.db not found" -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd"
$Archive = Join-Path $BackupDir "stock_history_$Stamp.tar.gz"

# Compress with relative path to avoid absolute entries in archive
Write-Host "[INFO] Compressing... (db ~2GB, takes 1-3 min)"
Push-Location $Root
try {
    tar -a -cf $Archive "data\stock_history.db"
} finally {
    Pop-Location
}

$SizeMB = "{0:N1}" -f ((Get-Item $Archive).Length / 1MB)
Write-Host "[OK] Compressed: $Archive ($SizeMB MB)" -ForegroundColor Green

# Upload
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $Tag = "data-backup-$Stamp"
    gh release create $Tag --repo $Repo --title "Data backup $Stamp" --notes "stock_history.db auto compressed backup" --clobber 2>$null
    if ($?) {
        gh release upload $Tag $Archive --repo $Repo --clobber
        if ($?) {
            Write-Host "[OK] Uploaded: https://github.com/$Repo/releases/tag/$Tag" -ForegroundColor Green
        } else {
            Write-Host "[WARN] Upload failed. Manual upload: https://github.com/$Repo/releases/new" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] Release create failed. Check network or gh auth" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] gh CLI not installed, skip auto-upload" -ForegroundColor Yellow
    Write-Host "       Manual: open https://github.com/$Repo/releases/new"
    Write-Host "       Drag $Archive into the page and publish."
}