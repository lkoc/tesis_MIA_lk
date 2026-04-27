#!/usr/bin/env python
"""Final blocking wait for 128×5 search completion."""
import time
import pathlib
import subprocess
from datetime import datetime

def check_search_complete():
    """Check if search completed by looking for BEST TRIAL in log."""
    log_path = pathlib.Path("examples/kim_2024_154kv_optim_C/optim_C_full14_fast.log")
    
    if not log_path.exists():
        return False, "Log file not yet created"
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        has_best_trial = "BEST TRIAL" in content
        lines = content.split('\n')
        
        # Count trials (look for "[ N/30]" pattern)
        trial_lines = [l for l in lines if '[' in l and '/30]' in l]
        n_trials = len(trial_lines)
        
        return has_best_trial and n_trials == 30, f"{n_trials}/30 trials, has_best={has_best_trial}"
    except Exception as e:
        return False, f"Error: {e}"

print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for 128×5 full 14-param search to complete...")
print("Checking every 30 seconds. ETA: 2.5-3 hours.")

max_wait = 14400  # 4 hours
check_interval = 30
elapsed = 0

while elapsed < max_wait:
    complete, status = check_search_complete()
    
    if complete:
        print(f"\n✓ SEARCH COMPLETE at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Status: {status}\n")
        
        # Extract and display results
        log_path = pathlib.Path("examples/kim_2024_154kv_optim_C/optim_C_full14_fast.log")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find and display results section
        for i, line in enumerate(lines):
            if "RESULTS" in line or "Rank" in line:
                print("".join(lines[i:min(i+60, len(lines))]))
                break
        
        exit(0)
    
    if elapsed % 300 == 0:  # Every 5 minutes
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} (waited {elapsed}s / {max_wait}s)")
    
    time.sleep(check_interval)
    elapsed += check_interval

print(f"\n⏱ Timeout after {elapsed} seconds")
complete, status = check_search_complete()
print(f"Last status: {status}")
