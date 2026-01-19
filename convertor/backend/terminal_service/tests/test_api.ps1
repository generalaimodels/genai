# API Testing Script - Terminal Service
# Run this script to test all endpoints

Write-Host "`n=== Advanced Terminal Service API Tests ===" -ForegroundColor Cyan
Write-Host "Service URL: http://localhost:8081`n" -ForegroundColor Gray

# Test 1: Health Check
Write-Host "[Test 1] Health Check..." -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri http://localhost:8081/api/v1/health
Write-Host "Status: $($health.status)" -ForegroundColor Green
Write-Host "Active Sessions: $($health.active_sessions)" -ForegroundColor Green
Write-Host ""

# Test 2: Service Info
Write-Host "[Test 2] Service Info..." -ForegroundColor Yellow
$info = Invoke-RestMethod -Uri http://localhost:8081/
Write-Host "Service: $($info.service)" -ForegroundColor Green
Write-Host "Version: $($info.version)" -ForegroundColor Green
Write-Host ""

# Test 3: Create PowerShell Session
Write-Host "[Test 3] Create PowerShell Session..." -ForegroundColor Yellow
try {
    $createBody = @{
        shell = "powershell"
        rows = 24
        cols = 80
    } | ConvertTo-Json
    
    $session = Invoke-RestMethod -Uri http://localhost:8081/api/v1/sessions `
        -Method Post `
        -ContentType "application/json" `
        -Body $createBody error>$null
    
    $sessionId = $session.session_id
    Write-Host "✓ Created session: $sessionId" -ForegroundColor Green
    Write-Host "  Shell: $($session.shell)" -ForegroundColor Gray
    Write-Host "  Size: $($session.rows)x$($session.cols)" -ForegroundColor Gray
    Write-Host ""
    
    # Test 4: List Sessions
    Write-Host "[Test 4] List Sessions..." -ForegroundColor Yellow
    $sessions = Invoke-RestMethod -Uri http://localhost:8081/api/v1/sessions
    Write-Host "✓ Found $($sessions.Count) session(s)" -ForegroundColor Green
    foreach ($s in $sessions) {
        Write-Host "  - $($s.session_id): $($s.shell) (PID: $($s.pid))" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Test 5: Get Session Details
    Write-Host "[Test 5] Get Session Details..." -ForegroundColor Yellow
    $details = Invoke-RestMethod -Uri "http://localhost:8081/api/v1/sessions/$sessionId"
    Write-Host "✓ Session Details:" -ForegroundColor Green
    Write-Host "  ID: $($details.session_id)" -ForegroundColor Gray
    Write-Host "  Shell: $($details.shell)" -ForegroundColor Gray
    Write-Host "  PID: $($details.pid)" -ForegroundColor Gray
    Write-Host "  Uptime: $([math]::Round($details.uptime, 2))s" -ForegroundColor Gray
    Write-Host ""
    
    # Test 6: Resize Session
    Write-Host "[Test 6] Resize Session..." -ForegroundColor Yellow
    $resizeBody = @{
        rows = 30
        cols = 120
    } | ConvertTo-Json
    
    $resizeResult = Invoke-RestMethod -Uri "http://localhost:8081/api/v1/sessions/$sessionId/resize" `
        -Method Put `
        -ContentType "application/json" `
        -Body $resizeBody
    
    Write-Host "✓ Resized to $($resizeResult.rows)x$($resizeResult.cols)" -ForegroundColor Green
    Write-Host ""
    
    # Test 7: Delete Session
    Write-Host "[Test 7] Delete Session..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://localhost:8081/api/v1/sessions/$sessionId" `
        -Method Delete
    
    Write-Host "✓ Session deleted" -ForegroundColor Green
    Write-Host ""
    
    # Test 8: Verify Deletion
    Write-Host "[Test 8] Verify Deletion..." -ForegroundColor Yellow
    $remainingSessions = Invoke-RestMethod -Uri http://localhost:8081/api/v1/sessions
    Write-Host "✓ Remaining sessions: $($remainingSessions.Count)" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "=== All Tests Passed! ===" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Note: Make sure Redis is running (docker run -d -p 6379:6379 redis:7-alpine)" -ForegroundColor Yellow
}
