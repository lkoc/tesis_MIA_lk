"""Dimensional validation for the quantities that enter the boundary and contact data.

Temperatures are in degrees Celsius; every other quantity is SI. A bare number is read
in the canonical unit of its role; a {"value", "unit"} dictionary is checked against it.
Thermal conductivity k (W/(m K)) is never confused with thermal resistivity (K m/W).
"""
import math
from pinn_v2.cases.errors import DimensionalError

CANONICAL = {
    'temperature': 'degC',
    'conductivity': 'W/(m K)',
    'heat_transfer_coefficient': 'W/(m2 K)',
    'heat_flux': 'W/m2',
    'contact_resistance': 'm2 K/W',
    'thermal_resistivity': 'K m/W',
}
_ACCEPTED = {
    'temperature': {'degc', 'c'},
    'conductivity': {'w/mk', 'wm-1k-1'},
    'heat_transfer_coefficient': {'w/m2k', 'wm-2k-1'},
    'heat_flux': {'w/m2', 'wm-2'},
    'contact_resistance': {'m2k/w', 'km2/w'},
    'thermal_resistivity': {'km/w', 'mk/w'},
}


def normalize_unit(unit):
    text = str(unit).lower()
    for char in ' ()*^.' + chr(183) + chr(176) + '_':
        text = text.replace(char, '')
    return text


def kind_of_unit(unit):
    norm = normalize_unit(unit)
    for kind, accepted in _ACCEPTED.items():
        if norm in accepted:
            return kind
    return None


def quantity(value, kind, label):
    """Return the float in the canonical unit of `kind`, rejecting a wrong unit."""
    if kind not in CANONICAL:
        raise ValueError(f'Unknown quantity kind {kind}')
    if isinstance(value, dict):
        if set(value) != {'value', 'unit'}:
            raise DimensionalError(f'{label}: a quantity needs exactly value and unit')
        unit = value['unit']
        actual = kind_of_unit(unit)
        if actual != kind:
            hint = ''
            if kind == 'conductivity' and actual == 'thermal_resistivity':
                hint = ' (thermal resistivity K m/W is not conductivity; use k = 1/rho explicitly)'
            raise DimensionalError(f'{label}: unit {unit!r} is not {CANONICAL[kind]}{hint}')
        value = value['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DimensionalError(f'{label}: finite number required')
    return float(value)


def conductivity_from_resistivity(rho_K_m_W):
    if not math.isfinite(rho_K_m_W) or rho_K_m_W <= 0:
        raise DimensionalError('Thermal resistivity must be positive and finite')
    return 1.0 / rho_K_m_W