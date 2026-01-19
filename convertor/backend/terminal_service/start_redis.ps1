# Start Redis via WSL - Quick Helper Script
# This starts Redis in WSL if available

Write-Host "`n=== Starting Redis via WSL ===" -ForegroundColor Cyan

try {
    # Check if WSL is available
    $wslCheck = wsl --list --quiet 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ WSL detected" -ForegroundColor Green
        
        # Install Redis if not present
        Write-Host "Installing Redis (if needed)..." -ForegroundColor Yellow
        wsl bash -c "sudo apt-get update > /dev/null 2>&1 && sudo apt-get install -y redis-server > /dev/null 2>&1"
        
        # Start Redis
        Write-Host "Starting Redis server..." -ForegroundColor Yellow
        wsl bash -c "redis-server --daemonize yes --bind 0.0.0.0 2>&1"
        
        Start-Sleep -Seconds 2
        
        # Test connection
        $pingResult = wsl bash -c "redis-cli ping 2>&1"
        
        if ($pingResult -like "*PONG*") {
            Write-Host "✓ Redis is running successfully!" -ForegroundColor Green
            Write-Host "`nYou can now run the TUI demo:" -ForegroundColor Cyan
            Write-Host "  python tests/interactive_tui.py" -ForegroundColor White
        }
        else {
            Write-Host "✗ Redis started but not responding" -ForegroundColor Red
            Write-Host "Output: $pingResult" -ForegroundColor Gray
        }
    }
    else {
        Write-Host "✗ WSL not available" -ForegroundColor Red
        Write-Host "`nAlternative: Download Redis for Windows" -ForegroundColor Yellow
        Write-Host "https://github.com/tporadowski/redis/releases" -ForegroundColor Gray
    }
}
catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
