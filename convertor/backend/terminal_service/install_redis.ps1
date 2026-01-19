# Quick Redis Installation and Start Script
# This will download and run Redis in a simple way for testing

Write-Host "=== Installing Redis for Windows ===" -ForegroundColor Cyan

# Check if winget is available
if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "Installing Redis using winget..." -ForegroundColor Yellow
    winget install Redis.Redis
}
else {
    Write-Host "Winget not found. Please install Redis manually:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Download from https://github.com/microsoftarchive/redis/releases" -ForegroundColor White
    Write-Host "Option 2: Use Docker (if available): docker run -d -p 6379:6379 redis:7-alpine" -ForegroundColor White
    Write-Host "Option 3: Use WSL: wsl sudo apt install redis-server && wsl redis-server" -ForegroundColor White
    Write-Host ""
}

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
