"""PINN_v2 copies must reproduce the legacy physics exactly; originals stay untouched."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

TOP=Path(__file__).resolve().parents[1]
REPO=TOP.parent

if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from pinn_v2.cases.catalog import cases as new_cases
from pinn_v2.geometry.domain import FullDomain as NewDomain, coaxial_case as new_coaxial


def legacy_modules():
    try:
        from Benchmarks.cases import cases as old_cases
        from Benchmarks.full_domain import FullDomain as OldDomain, coaxial_case as old_coaxial
    except Exception as error:
        pytest.skip(f'legacy Benchmarks not importable: {error}')
    return old_cases,OldDomain,old_coaxial


def test_catalog_has_the_same_nineteen_cases_and_values():
    old_cases,_,_=legacy_modules()
    old=old_cases();new=new_cases()
    assert len(new)==19
    assert old==new


def test_fingerprint_and_areas_match_for_every_cable_case():
    old_cases,OldDomain,_=legacy_modules()
    old=old_cases();checked=0
    for name,case in new_cases().items():
        if case['kind']!='cable':continue
        mode='fixed'
        a=OldDomain(old[name],mode);b=NewDomain(case,mode)
        assert a.fingerprint==b.fingerprint,name
        assert [r.name for r in a.regions]==[r.name for r in b.regions]
        assert [a.area(r) for r in a.regions]==[b.area(r) for r in b.regions]
        for r in b.regions:
            assert np.array_equal(a.sample(a.regions[r.id],64,7+r.id),b.sample(r,64,7+r.id))
        checked+=1
    assert checked>=8


def test_coaxial_fingerprint_matches_and_is_deterministic():
    _,OldDomain,old_coaxial=legacy_modules()
    assert OldDomain(old_coaxial()).fingerprint==NewDomain(new_coaxial()).fingerprint
    a=NewDomain(new_coaxial());b=NewDomain(new_coaxial())
    xa,la,wa=a.evaluation_points();xb,lb,wb=b.evaluation_points()
    assert np.array_equal(xa,xb) and np.array_equal(la,lb) and np.array_equal(wa,wb)


def test_geometry_and_cases_do_not_import_torch():
    code='import sys;import pinn_v2.geometry.domain,pinn_v2.cases.catalog,pinn_v2.manifest.provenance;assert "torch" not in sys.modules,"torch leaked"'
    done=subprocess.run([sys.executable,'-c',code],cwd=TOP,capture_output=True,text=True)
    assert done.returncode==0,done.stderr


def test_provenance_record_matches_untouched_originals():
    record=json.loads((TOP/'COPIED_FROM.json').read_text(encoding='utf-8'))
    assert len(record)==29
    for item in record:
        original=REPO/item['source']
        assert original.is_file(),item['source']
        assert hashlib.sha256(original.read_bytes()).hexdigest()==item['sha256'],f"{item['source']} was modified"
        assert (TOP/item['destination']).is_file()


def test_no_residual_references_to_legacy_packages():
    for path in (TOP/'pinn_v2').rglob('*.py'):
        text=path.read_text(encoding='utf-8')
        assert 'from Benchmarks' not in text and 'import Benchmarks' not in text,path
        assert 'pinn_cables' not in text.replace('vendored from pinn_cables',''),path
