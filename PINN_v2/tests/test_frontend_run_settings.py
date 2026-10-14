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