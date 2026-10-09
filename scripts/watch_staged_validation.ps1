param([switch]$Once)
$projectRoot = Split-Path $PSScriptRoot -Parent
$runRoot = Join-Path $projectRoot 'artifacts\staged_validation'
$previousStage = ''
$previousProgress = ''
$logLines = @{}
Write-Host 'Live validation and training monitor. Ctrl+C stops only this monitor.' -ForegroundColor Cyan
while ($true) {
    try {
        $status = Get-Content -LiteralPath (Join-Path $runRoot 'status.json') -Raw -ErrorAction Stop | ConvertFrom-Json
        if ($status.stage -ne $previousStage) {
            Write-Host ("`n[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $status.stage) -ForegroundColor Cyan
            $previousStage = $status.stage
            $previousProgress = ''
        }
        $progressPath = switch ($status.stage) {
            'waiting_for_nominal' { Join-Path $projectRoot 'artifacts\feasibility_baseline_optimized\exact_state_baseline_live\progress.json' }
            'running_nominal' { Join-Path $runRoot 'nominal\exact_state_baseline_live\progress.json' }
            'running_disturbance_tests' { Join-Path $runRoot 'disturbed\exact_state_baseline_live\progress.json' }
            'running_ppo_pilot' { Join-Path $runRoot 'pilot_checkpoints\baseline_diagnostics\progress.json' }
        }
        if ($progressPath -and (Test-Path -LiteralPath $progressPath)) {
            $raw = Get-Content -LiteralPath $progressPath -Raw
            if ($raw -ne $previousProgress) {
                $progress = $raw | ConvertFrom-Json
                Write-Host ("Flight: {0} | wall {1:N0}s" -f $progress.status, $progress.wall_seconds)
                foreach ($trial in $progress.trials) {
                    Write-Host ("  Trial {0} | sim {1:N2}s | gates {2}/{3} | error {4:N4}m" -f ($trial.trial + 1), $trial.simulated_seconds, $trial.gates_passed, $trial.total_gates, $trial.tracking_error_m)
                    Write-Host ("  collision {0} | fallbacks {1} | success {2}" -f $trial.collision, $trial.fallbacks, $trial.success)
                }
                $previousProgress = $raw
            }
        }
        foreach ($file in @(Get-ChildItem -LiteralPath $runRoot -Filter '*.log' -ErrorAction SilentlyContinue)) {
            $lines = @(Get-Content -LiteralPath $file.FullName)
            $seen = if ($logLines.ContainsKey($file.FullName)) { $logLines[$file.FullName] } else { [Math]::Max(0, $lines.Count - 20) }
            if ($seen -gt $lines.Count) { $seen = 0 }
            if ($lines.Count -gt $seen) {
                Write-Host ("--- {0} ---" -f $file.Name) -ForegroundColor DarkCyan
                for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Host $lines[$i] }
            }
            $logLines[$file.FullName] = $lines.Count
        }
        if ($status.stage -in @('completed', 'blocked', 'interrupted')) {
            if ($status.reason) { Write-Host $status.reason -ForegroundColor Yellow }
            if ($status.checkpoint) { Write-Host ("Checkpoint: " + $status.checkpoint) }
            break
        }
    } catch {
        Write-Host ("Waiting for a readable status update: " + $_.Exception.Message) -ForegroundColor DarkYellow
    }
    if ($Once) { break }
    Start-Sleep -Seconds 2
}
