import pytest

from tools.medium_fem_init_smoke import cases_from


def test_campaign_resume_can_start_at_aras_flat_without_reordering_cases():
    cases = ('mms_robin', 'aras_flat', 'aras_single', 'cable_disk')

    assert cases_from(cases, 'aras_flat') == ('aras_flat', 'aras_single', 'cable_disk')
    assert cases == ('mms_robin', 'aras_flat', 'aras_single', 'cable_disk')


def test_campaign_resume_rejects_unknown_start_case():
    with pytest.raises(ValueError, match='not in the campaign case list'):
        cases_from(('aras_flat', 'cable_disk'), 'missing_case')