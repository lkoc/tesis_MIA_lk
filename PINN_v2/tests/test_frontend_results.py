import json
import importlib.util
import sys
from pathlib import Path

import pytest

PINN_V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PINN_V2 / 'front-end'))
STORE_SPEC = importlib.util.spec_from_file_location('store', PINN_V2 / 'front-end' / 'store.py')
assert STORE_SPEC is not None and STORE_SPEC.loader is not None
store = importlib.util.module_from_spec(STORE_SPEC)
sys.modules[STORE_SPEC.name] = store
STORE_SPEC.loader.exec_module(store)


def write_fem(root, name, case_id='kim_2024_thermal_original'):
    root.mkdir(parents=True, exist_ok=True)
    record = {
        'specification': {'case': {'id': case_id, 'kind': 'cable'}},
        'level': 0,
        'Tmax_C': 78.2,
        'ndofs': 172145,
        'physics_sha256': 'same-physics',
    }
    root.joinpath(name + '.json').write_text(json.dumps(record), encoding='utf-8')
    root.joinpath(name + '.npz').write_bytes(b'present')


def test_kim_benchmark_result_is_listed_read_only_and_path_scoped(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    external = tmp_path / 'Benchmarks' / 'explicit_study' / 'D' / 'kim_thermal_shared_20261002' / 'fem_reaction_balance'
    write_fem(external, 'fem_l0')
    write_fem(tmp_path / 'other' / 'fem_l1', 'fem_l1', 'not-allowlisted')
    monkeypatch.setattr(store, 'RESULTS', results)
    monkeypatch.setattr(store, 'EXTERNAL_RESULTS', external)

    items = store.scan_results()

    assert len(items) == 1
    assert items[0]['sid'] == 'res:kim_thermal/fem_l0'
    assert items[0]['case'] == 'kim_2024_thermal_original'
    assert items[0]['origin'] == 'benchmarks'
    assert 'smoke, no convergido' in items[0]['label']
    assert store._result_path(items[0]['sid']) == external.joinpath('fem_l0').resolve()
    with pytest.raises(KeyError, match='ruta de resultado'):
        store._result_path('res:kim_thermal/../outside/fem_l0')
    with pytest.raises(KeyError, match='ruta de resultado'):
        store._result_path('res:')