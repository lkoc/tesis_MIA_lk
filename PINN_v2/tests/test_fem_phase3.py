"""Phase 3: reference FEM contract (manifests, no-overwrite, Neumann/Robin, V0-V2 convergence).

The pure-Python tests always run. Tests marked fem call the FEniCSx environment in WSL2 by subprocess
and are skipped when it is unavailable.
"""
import json
import math
from pathlib import Path
import pytest
from pinn_v2.cases.definition import load_all
from pinn_v2.cases.verification import coaxial_robin, symmetric_pair
from pinn_v2.experiments import fem_reference
from pinn_v2.manifest.run_manifest import build_manifest, write_json_atomic, write_manifest
from pinn_v2.solvers.fem import adapter

needs_fem = pytest.mark.skipif(not adapter.fem_available(), reason='FEniCSx environment not available in WSL2 Ubuntu')


def test_write_json_atomic_never_overwrites(tmp_path):
    target = tmp_path.joinpath('a', 'record.json')
    write_json_atomic(target, dict(value=1))
    with pytest.raises(FileExistsError):
        write_json_atomic(target, dict(value=2))
    assert json.loads(target.read_text(encoding='utf-8')) == dict(value=1)
    assert [p.name for p in target.parent.iterdir()] == ['record.json']


def test_manifest_records_inputs_seeds_environment_and_sources(tmp_path):
    source = tmp_path.joinpath('input.json')
    source.write_text('{}', encoding='utf-8')
    manifest = build_manifest('test', ['cmd'], inputs=[source], seeds=dict(train=1, evaluation=2), fem_environment=dict(dolfinx='x'))
    path = write_manifest(tmp_path, manifest)
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['inputs']['input.json'] and saved['seeds'] == dict(train=1, evaluation=2)
    assert saved['fem_environment'] == dict(dolfinx='x') and saved['git']['commit'] is not None
    assert any(name.endswith('full_fem.py') for name in saved['source_sha256'])
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path, manifest)


def test_study_refuses_an_existing_directory(tmp_path):
    tmp_path.joinpath('results', 'phase3', 'old').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        fem_reference.run_study('old', root=tmp_path)


def test_observed_order_recovers_a_known_rate():
    sizes = [.1, .05, .025]
    assert fem_reference.observed_orders([s ** 3 for s in sizes], sizes) == pytest.approx([3., 3.])


def test_wsl_path_maps_windows_drives():
    assert adapter.wsl_path(Path('C:/usr/x/y.json')) == '/mnt/c/usr/x/y.json'


def test_verification_definitions_keep_robin_and_symmetry_neumann():
    robin = coaxial_robin(15., 20.).domain()
    assert robin.side_conditions()['outer']['type'] == 'robin' and not robin.all_dirichlet
    half = symmetric_pair(True).domain()
    assert half.side_conditions()['left'] == dict(type='neumann', q_n_W_m2=0.)
    assert half.side_conditions()['top']['type'] == 'robin'
    full = symmetric_pair(False).domain()
    assert len(full.conductors) == 2 and len(half.conductors) == 1
    assert math.isclose(half.source_integral(half.conductors[0]), full.source_integral(full.conductors[0]), rel_tol=1e-9)


@pytest.mark.fem
@needs_fem
@pytest.mark.parametrize('case', ['mms_constant', 'mms_interface', 'mms_robin'])
def test_manufactured_reference_converges_at_p2_rate(tmp_path, case):
    records = adapter.run_mms_fem(tmp_path.joinpath(case), case, [0, 1, 2])
    errors = [r['l2_error_K'] for r in records]
    orders = fem_reference.observed_orders(errors, [r['h'] for r in records])
    assert all(o > 2.8 for o in orders), orders
    assert errors[-1] < 1e-4
    assert all(any(name.endswith('mms_fem.py') for name in r['source_sha256']) for r in records)
    if case == 'mms_interface':
        jumps = [r['interface_flux_jump_rel'] for r in records]
        assert jumps[0] > jumps[1] > jumps[2] and jumps[2] < 1e-3
    with pytest.raises(RuntimeError):
        adapter.run_mms_fem(tmp_path.joinpath(case), case, [0])


@pytest.mark.fem
@needs_fem
def test_robin_coaxial_matches_radial_solution_and_closes_the_balance(tmp_path):
    definition = coaxial_robin()
    path = tmp_path.joinpath('definition.json')
    write_json_atomic(path, definition.to_dict())
    records = adapter.run_cable_fem(tmp_path.joinpath('out'), [0, 1], definition=path)
    assert all(r['rmse_exact_K'] < 5e-3 and r['error_Tmax_exact_K'] < 5e-3 for r in records)
    assert all(r['balance_consistent_pct'] < 1e-5 and r['balance_pct'] < 1. for r in records)
    assert records[1]['rmse_exact_K'] < records[0]['rmse_exact_K']
    check = records[1]['boundary_check']['outer'] if 'outer' in records[1]['boundary_check'] else None
    assert check is None or check['type'] == 'robin'
    assert records[1]['boundary_conditions']['outer']['type'] == 'robin'


@pytest.mark.fem
@needs_fem
def test_symmetry_side_is_equivalent_to_the_full_domain(tmp_path):
    full, half = symmetric_pair(False), symmetric_pair(True)
    results = {}
    for definition in (full, half):
        path = tmp_path.joinpath(definition.id + '.json')
        write_json_atomic(path, definition.to_dict())
        results[definition.id] = adapter.run_cable_fem(tmp_path.joinpath(definition.id), [1], definition=path)[0]
    f, h = results['symmetric_full'], results['symmetric_half']
    assert f['source_W_m'] == pytest.approx(2 * h['source_W_m'], rel=1e-9)
    ratio = f['boundary_check']['top']['constitutive_outflow_W_m'] / h['boundary_check']['top']['constitutive_outflow_W_m']
    assert ratio == pytest.approx(2., rel=5e-3)
    assert abs(h['boundary_check']['left']['traction_outflow_W_m']) < 0.05 * h['source_W_m']
    assert h['balance_consistent_pct'] < 1e-5 and f['balance_consistent_pct'] < 1e-5
    full_means = [r['T_mean_C'] for r in f['regions'] if 'T_mean_C' in r]
    half_mean = [r['T_mean_C'] for r in h['regions'] if 'T_mean_C' in r][0]
    assert half_mean == pytest.approx(full_means[1], abs=0.05)


@pytest.mark.fem
@needs_fem
def test_fem_refuses_imperfect_contact_instead_of_silently_using_perfect_contact(tmp_path):
    data = load_all()['xlpe_single'].to_dict()
    data['contact'] = [dict(interface='cable_0_interface_1', R=dict(value=2e-3, unit='m2 K/W'), status='supuesto', source=data['provenance'][0]['id'])]
    data['data_status']['contact'] = dict(status='supuesto', source=data['provenance'][0]['id'], note='test value, not a measurement')
    path = tmp_path.joinpath('contact.json')
    path.write_text(json.dumps(data), encoding='utf-8')
    done = adapter.run_module('pinn_v2.solvers.fem.full_fem', ['--definition', path, '--levels', '0', '--output', tmp_path.joinpath('out')], check=False)
    assert done.returncode != 0 and 'NotImplementedError' in done.stderr
    assert not tmp_path.joinpath('out', 'fem_l0.json').exists()