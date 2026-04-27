#!/usr/bin/env python
"""Wait for 128×5 full search completion and retrieve results."""
import time
import pathlib
import subprocess
from datetime import datetime

log_path = pathlib.Path("examples/kim_2024_154kv_optim_C/optim_C_full14_fast.log")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring: {log_path.absolute()}")

max_wait = 14400  # 4 hours
check_interval = 60  # Check every 60 seconds
elapsed = 0

while elapsed < max_wait:
    try:
        # Check if process is still running
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FI", "STATUS eq running"],
            capture_output=True, text=True
        )
        has_python = "python.exe" in result.stdout
        
        if log_path.exists():
            log_size = log_path.stat().st_size
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Count completed trials
            trial_lines = [l for l in lines if l.strip().startswith('[')]
            n_completed = len(trial_lines)
            
            # Check for final summary
            has_summary = any("BEST TRIAL" in l for l in lines)
            
            if n_completed > 0 or log_size > 1000:
                if elapsed % 300 == 0:  # Print every 5 minutes
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {n_completed} trials, {log_size} bytes, process_active={has_python}")
            
            # If we have summary, we're done
            if has_summary and not has_python:
                print(f"\n✓ Search complete! Extracting results...")
                # Get last 80 lines (should include full results table + best trial)
                summary_lines = lines[-80:]
                print("".join(summary_lines))
                break
        
        if not has_python and log_path.exists():
            # Process ended, get final results
            with open(log_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 50:
                print(f"\n✓ Process complete. Final output:")
                print("".join(lines[-100:]))
                break
    
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
    
    time.sleep(check_interval)
    elapsed += check_interval
    
    if elapsed > max_wait:
        print(f"\n⏱ Timeout after {elapsed} seconds")
        if log_path.exists():
            print("Last 50 lines of log:")
            with open(log_path, 'r') as f:
                lines = f.readlines()
            print("".join(lines[-50:]))
