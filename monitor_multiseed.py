#!/usr/bin/env python
"""Monitor multi-seed training and collect final results."""
import time
import pathlib
import json
from datetime import datetime

model_b_path = pathlib.Path("examples/kim_2024_154kv_optim_B/results_corrected_optim/model_best_64x4.pt")
model_c_path = pathlib.Path("examples/kim_2024_154kv_optim_C/results_corrected_optim/model_best_128x5.pt")

print("[Monitor] Waiting for model files...")
print(f"  64×4: {model_b_path.absolute()}")
print(f"  128×5: {model_c_path.absolute()}")
print()

max_wait = 14400  # 4 hours
check_interval = 60  # Check every 60 seconds
elapsed = 0

while elapsed < max_wait:
    b_exists = model_b_path.exists()
    c_exists = model_c_path.exists()
    
    if b_exists:
        print(f"✓ 64×4 model found! Size: {model_b_path.stat().st_size / (1024*1024):.1f} MB")
    if c_exists:
        print(f"✓ 128×5 model found! Size: {model_c_path.stat().st_size / (1024*1024):.1f} MB")
    
    if b_exists and c_exists:
        print(f"\n✓ Both models ready after {elapsed} seconds ({elapsed/60:.1f} minutes)")
        
        # Try to load and extract metadata
        try:
            import torch
            checkpoint_b = torch.load(model_b_path, map_location='cpu')
            print(f"\n64×4 Results:")
            print(f"  T_max: {checkpoint_b.get('T_max_C', 'N/A'):.2f} °C")
            print(f"  Error: {checkpoint_b.get('error_K', 'N/A'):.2f} K")
            print(f"  Seed: {checkpoint_b.get('seed', 'N/A')}")
            
            checkpoint_c = torch.load(model_c_path, map_location='cpu')
            print(f"\n128×5 Results:")
            print(f"  T_max: {checkpoint_c.get('T_max_C', 'N/A'):.2f} °C")
            print(f"  Error: {checkpoint_c.get('error_K', 'N/A'):.2f} K")
            print(f"  Seed: {checkpoint_c.get('seed', 'N/A')}")
        except Exception as e:
            print(f"Could not load checkpoints: {e}")
        
        break
    
    if elapsed % 300 == 0:  # Print every 5 minutes
        remaining_hrs = (max_wait - elapsed) / 3600
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Still waiting... ({remaining_hrs:.1f} hours remaining)")
        print(f"  64×4: {'✓ ready' if b_exists else '⏳ training...'}")
        print(f"  128×5: {'✓ ready' if c_exists else '⏳ training...'}")
    
    time.sleep(check_interval)
    elapsed += check_interval

if not (model_b_path.exists() and model_c_path.exists()):
    print(f"\n⏱ Timeout after {elapsed} seconds. Models not yet ready.")
    print("Training still in progress. Check back later for results.")
