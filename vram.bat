@echo off
powershell -NoProfile -Command ^
  "$lines = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.CookedValue -gt 10485760 } | Sort-Object CookedValue -Descending | Select-Object -First 20; foreach ($l in $lines) { $l.InstanceName -match 'pid_(\d+)' | Out-Null; $p = $Matches[1]; $mb = [math]::Round($l.CookedValue/1MB); $name = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName; if (-not $name) { $name = '???' }; Write-Output ('{0,6} MB  {1}' -f $mb, $name) }"
