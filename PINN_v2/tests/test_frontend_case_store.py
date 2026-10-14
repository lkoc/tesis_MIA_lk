import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'front-end'))

import casestore
from pinn_v2.cases.catalog import cases
from pinn_v2.cases.definition import CaseDefinition


def test_catalog_discovers_nested_v2_json_and_loads_relative_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(casestore, 'V2_DIR', tmp_path / 'v2')
    monkeypatch.setattr(casestore, 'USER', tmp_path / 'user_cases')
    path = casestore.V2_DIR / 'papers' / 'aras' / 'aras_single.json'
    path.parent.mkdir(parents=True)
    document = CaseDefinition.from_legacy(cases()['aras_single'], 'test', 'unit test').to_dict()
    path.write_text(json.dumps(document), encoding='utf-8')

    listing = casestore.listing()
    assert len(listing['catalog']) == 1
    item = listing['catalog'][0]
    assert item['ref'] == 'catalog:v2:papers/aras/aras_single'
    assert item['folder'] == 'papers/aras'
    assert casestore.load(item['ref'])['doc']['id'] == 'aras_single'
    assert all(entry['group'] == 'v2' for entry in listing['catalog'])


def test_v2_catalog_reference_rejects_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(casestore, 'V2_DIR', tmp_path / 'v2')
    with pytest.raises(ValueError, match='referencia v2 no valida'):
        casestore.load('catalog:v2:../outside')


def test_legacy_catalog_references_are_not_loadable():
    with pytest.raises(KeyError, match='caso no encontrado'):
        casestore.load('catalog:legacy:xlpe_single')


def test_every_active_case_passes_frontend_pinn_and_fem_support_checks():
    listing = casestore.listing()
    assert listing['catalog']
    for entry in listing['catalog']:
        verdict = casestore.check(casestore.load(entry['ref'])['doc'])
        assert verdict['ok'], (entry['ref'], verdict['errors'])
        assert verdict['pinn']['supported'], (entry['ref'], verdict['pinn']['reason'])
        assert verdict['fem']['supported'], (entry['ref'], verdict['fem']['reason'])


def test_verification_grids_preserve_mms_regions_and_annulus_hole(tmp_path):
    import grids
    from pinn_v2.solvers.pinn.problem import load_problem

    layered = load_problem('mms_layered_y')
    cut = layered.regions[0].upper
    X, Y = np.meshgrid([0.5], [cut / 2, (cut + 1) / 2])
    assert grids.label_problem(layered, X, Y).ravel().tolist() == [0, 1]

    annulus = load_problem('annulus')
    from pinn_v2.solvers.pinn.networks import build_model
    model = build_model(annulus, 'M1', 8, 2, 'tanh').double()
    pinn_pieces = grids.build_problem_pinn(model, annulus, full_n=32, zoom_n=32)
    pinn_labels = pinn_pieces['full']['L']
    assert np.any(pinn_labels == 0) and np.isfinite(pinn_pieces['full']['T'][pinn_labels == 0]).any()
    assert np.isnan(pinn_pieces['full']['T'][pinn_labels < 0]).all()

    xy, _, _ = annulus.evaluation()
    fem_path = tmp_path / 'annulus.npz'
    np.savez_compressed(fem_path, xy=xy, T=annulus.reference(xy))
    pieces = grids.build_verification_fem(fem_path, annulus, full_n=32, zoom_n=32)
    labels, temperatures = pieces['full']['L'], pieces['full']['T']
    assert np.any(labels == 0) and np.isfinite(temperatures[labels == 0]).any()
    assert np.isnan(temperatures[labels < 0]).all()


def test_evaluate_pinn_uses_model_device_and_returns_numpy():
    import torch
    import grids

    class DeviceCheckingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty((), device='meta'))
            self.input_devices = None

        def temperature(self, xy, labels):
            self.input_devices = (xy.device, labels.device)
            return torch.ones((len(xy), 1), dtype=xy.dtype)

    model = DeviceCheckingModel()
    coordinates = np.array([[0.0]])
    result = grids.evaluate_pinn(model, np.array([[0]]), coordinates, coordinates)

    assert model.input_devices == (torch.device('meta'), torch.device('meta'))
    assert np.array_equal(result, np.array([[1.0]]))