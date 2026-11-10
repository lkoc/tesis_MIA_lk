#!/usr/bin/env python
"""Quick test of multi-seed execution."""
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    print("Importing pinn_cables...")
    import pinn_cables
    print("  ✓ pinn_cables imported")
    
    print("Importing run_optim_B_corrected...")
    from examples.kim_2024_154kv_optim_B import run_optim_B_corrected
    print("  ✓ run_optim_B_corrected imported")
    
    print("Creating args...")
    import argparse
    args = argparse.Namespace(best=True, n_seeds=5, trial_seed=0, timeout=900, 
                              trials=30, fast=False, seed=0)
    print(f"  ✓ args created: {args}")
    
    print("Calling main()...")
    run_optim_B_corrected.main()
    print("  ✓ main() completed")
    
except Exception as e:
    import traceback
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
