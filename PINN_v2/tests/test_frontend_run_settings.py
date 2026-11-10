import json
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PINN_V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PINN_V2 / 'front-end'))
import server


def test_collocation_preview_reports_total_and_point_groups():
    case_path = PINN_V2 / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json'
    doc = json.loads(case_path.read_text(encoding='utf-8'))
    params = dict(architecture='M2', sampling_seed=11011,
        points=dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128))

    summary = server.collocation_preview(dict(doc=doc, params=params))
    preview = server.collocation_preview(dict(doc=doc, params=params, include_points=True))

    assert summary['summary'] == preview['summary']
    assert summary['summary']['soil_regions'] == 1
    assert summary['summary']['layer_regions'] == 12
    assert summary['summary']['interfaces'] == 12
    assert summary['summary']['boundary_pieces'] == 4
    assert summary['summary']['interior_points'] == 3584
    assert summary['summary']['interface_points'] == 1536
    assert summary['summary']['boundary_points'] == 512
    assert summary['summary']['total_points'] == 5632
    assert len(preview['points']['interior']) == 13
    assert sum(len(region['xy']) for region in preview['points']['interior']) == 3584


def test_field_png_renders_collocation_overlay():
    axis = np.linspace(0.1, 0.9, 4)
    piece = dict(x=axis, y=axis, T=np.full((4, 4), 25.), L=np.zeros((4, 4)), window=[0., 1., 0., 1.])
    png = server.render.field_png(dict(full=piece, zoom=piece), dict(kind='mms_constant', bounds=[0., 1., 0., 1.]),
        overlay=False, collocation=dict(interior=[np.array([[.2, .3]])], interfaces=[np.array([[.5, .5]])],
            boundary=[np.array([[.1, .1]])]))

    assert png.startswith(b'\x89PNG\r\n\x1a\n')


def test_collocation_preview_reports_case_total_and_exact_point_groups():
    case_path = PINN_V2 / 'pinn_v2' / 'cases' / 'v2' / 'references' / 'cables' / 'aras_flat.json'
    doc = json.loads(case_path.read_text(encoding='utf-8'))
    params = dict(architecture='M2', sampling_seed=11011,
        points=dict(n_interior=512, n_layer=256, n_face=128, n_boundary=128))

    summary = server.collocation_preview(dict(doc=doc, params=params))['summary']
    preview = server.collocation_preview(dict(doc=doc, params=params, include_points=True))

    assert summary == preview['summary']
    assert summary['soil_regions'] == 1 and summary['layer_regions'] == 12
    assert summary['interfaces'] == 12 and summary['boundary_pieces'] == 4
    assert summary['interior_points'] == 3584
    assert summary['interface_points'] == 1536
    assert summary['boundary_points'] == 512
    assert summary['total_points'] == 5632
    assert len(preview['points']['interior']) == 13
    assert sum(len(region['xy']) for region in preview['points']['interior']) == 3584


def test_run_settings_round_trip_by_case_ref(tmp_path, monkeypatch):
    path = tmp_path / 'run_settings.json'
    monkeypatch.setattr(server, 'RUN_SETTINGS', path)
    settings = dict(kind='pinn', arch='DeepXDE', preset='custom', deepxdeModel='PFNN',
                    p=dict(activation='sin', init_seed=42, weights=dict(interface=3.5),
                      points=dict(n_interior=8192)), threads=4, pointMultiplier=2,
                  femPreset='alto', nodeMultiplier=1.5, levels=[0, 1, 2])

    assert server.route_post('/api/run-settings', dict(ref='user:tests/case-a', settings=settings)) == dict(ok=True)

    assert server.route_get('/api/run-settings', {'ref': ['user:tests/case-a']}) == dict(settings=settings)


def test_frontend_offers_exact_interface_architecture():
    architectures = {item['id'] for item in server.options()['architectures']}
    assert 'M1_EXACT' in architectures
    assert 'DeepXDE_EXACT' in architectures
    assert {'M0_EXACT', 'DCSNN_EXACT', 'CUSP_EXACT', 'FBPINN_EXACT',
        'DeepXDE_PFNN_EXACT', 'DeepXDE_FNN_EXACT'} <= architectures


def test_frontend_import_exports_only_pair_and_infers_without_original_run(tmp_path, monkeypatch):
    import base64
    import io
    import zipfile
    from pinn_v2.solvers.pinn.trainer import run
    original = tmp_path / 'original'
    result = run('mms_constant', 'M1_EXACT', original, adam_steps=1, width=4, depth=1,
        points=dict(n_interior=6, n_layer=6, n_face=6, n_boundary=6), device='cpu', full_audit=False)
    assert result['status'] == 'completed'
    runs = tmp_path / 'jobs'
    monkeypatch.setattr(server, 'RUNS', runs)
    imported = server.route_post('/api/model/import', dict(
        specification=json.loads((original / 'model.json').read_text()),
        checkpoint_base64=base64.b64encode((original / 'model.pt').read_bytes()).decode('ascii')))
    prediction = server.route_post(f"/api/job/{imported['id']}/infer", dict(xy=[[.2, .3]], flux=True))
    assert len(prediction['T']) == 1 and len(prediction['q'][0]) == 2
    with zipfile.ZipFile(io.BytesIO(server.export_model_pair(imported['id']))) as archive:
        assert set(archive.namelist()) == {'model.pt', 'model.json'}
    assert imported['resume_available'] is True


@pytest.mark.parametrize('case_id', ['aras_flat', 'mms_interface', 'annulus'])
@pytest.mark.parametrize('architecture', ['M1_EXACT', 'DCSNN_EXACT', 'CUSP_EXACT', 'FBPINN_EXACT',
    'M0_EXACT', 'DeepXDE_PFNN_EXACT', 'DeepXDE_FNN_EXACT'])
def test_frontend_exact_mode_preflight_accepts_supported_problem_families(case_id, architecture, tmp_path, monkeypatch):
    if architecture.startswith('DeepXDE'):
        pytest.importorskip('deepxde')
    if case_id in ('aras_flat', 'annulus'):
        relative = ('cables/aras_flat.json' if case_id == 'aras_flat'
            else 'verification/annulus.json')
        case_path = PINN_V2 / 'pinn_v2' / 'cases' / 'v2' / 'references' / relative
        doc = json.loads(case_path.read_text(encoding='utf-8'))
    else:
        from pinn_v2.cases.catalog import cases
        from pinn_v2.cases.definition import CaseDefinition
        definition = CaseDefinition.from_legacy(cases()[case_id], 'test', 'MMS exact-boundary test')
        doc = definition.to_dict()

    monkeypatch.setattr(server, 'RUNS', tmp_path)
    monkeypatch.setattr(server, 'PIDS', {})
    monkeypatch.setattr(server, 'PIDS_PROCS', {})

    class FakeProcess:
        pid = 12345

    def fake_popen(*args, **kwargs):
        kwargs['stdout'].close()
        return FakeProcess()

    monkeypatch.setattr(server.subprocess, 'Popen', fake_popen)
    response = server.start_job(dict(kind='pinn', doc=doc, params=dict(architecture=architecture)))
    job = json.loads(tmp_path.joinpath(response['id'], 'job.json').read_text(encoding='utf-8'))
    assert job['params']['architecture'] == architecture


@pytest.mark.parametrize('params', [
    dict(levels=[0, 1], node_multiplier=1.5, refinement_preset='medio'),
    dict(levels=[3], node_multiplier=3),
])
def test_validate_fem_params_accepts_supported_controls(params):
    validated = server.validate_fem_params(params)

    assert validated['levels'] == params['levels']
    assert validated['node_multiplier'] == params.get('node_multiplier', 1.0)
    assert validated['refinement_preset'] == params.get('refinement_preset', 'manual')


@pytest.mark.parametrize('params', [
    dict(levels=[]),
    dict(levels=[0, 4]),
    dict(levels=[1, 1]),
    dict(node_multiplier=2.5),
    dict(refinement_preset='extremo'),
])
def test_validate_fem_params_rejects_unsupported_controls(params):
    with pytest.raises(ValueError):
        server.validate_fem_params(params)


@pytest.mark.parametrize('ref', ['', 'x' * 513])
def test_run_settings_reject_invalid_case_ref(ref):
    with pytest.raises(ValueError, match='referencia de caso no valida'):
        server.route_post('/api/run-settings', dict(ref=ref, settings={}))