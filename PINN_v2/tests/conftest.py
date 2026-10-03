"""Benchmark precision choices must not leak into unrelated package tests."""
import pytest
import torch


@pytest.fixture(autouse=True)
def restore_torch_process_settings():
    dtype=torch.get_default_dtype();threads=torch.get_num_threads();rng=torch.get_rng_state()
    try:yield
    finally:
        torch.set_default_dtype(dtype);torch.set_num_threads(threads);torch.set_rng_state(rng)
