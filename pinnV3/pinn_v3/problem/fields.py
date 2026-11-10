"""Spatially varying physical fields: numbers, algebraic expressions and 1D piecewise-linear tables."""
import math
import numpy as np
from pinn_v3.problem.expressions import evaluate


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_field(spec, where):
    """Check one field spec; expressions are parsed once at a test point."""
    if isinstance(spec, bool):
        raise ValueError(f'{where}: booleans are not fields')
    if isinstance(spec, (int, float)):
        _require(math.isfinite(spec), f'{where}: number must be finite')
        return
    _require(isinstance(spec, dict) and len(spec) == 1 and set(spec) <= {'expression', 'table'},
             f'{where}: use a number, {{"expression": ...}} or {{"table": ...}}')
    if 'expression' in spec:
        text = spec['expression']
        _require(isinstance(text, str) and text.strip(), f'{where}: expression must be a non-empty string')
        try:
            evaluate(text, np.array([0.5]), np.array([0.5]), np)
        except Exception as exc:
            raise ValueError(f'{where}: invalid expression {text!r}: {exc}') from exc
        return
    table = spec['table']
    _require(isinstance(table, dict) and set(table) == {'axis', 'coordinates', 'values'} and table['axis'] in ('x', 'y'),
             f'{where}: table needs axis x or y, coordinates and values')
    coordinates, values = table['coordinates'], table['values']
    _require(len(coordinates) >= 2 and len(coordinates) == len(values), f'{where}: table needs two or more matching entries')
    _require(all(a < b for a, b in zip(coordinates, coordinates[1:])), f'{where}: table coordinates must increase strictly')
    _require(all(math.isfinite(v) for v in list(coordinates) + list(values)), f'{where}: table values must be finite')


def evaluate_field(spec, x, y):
    """Field values at the points (x, y); tables interpolate linearly along their axis."""
    shape = np.shape(x)
    if isinstance(spec, (int, float)):
        return np.full(shape, float(spec))
    if 'expression' in spec:
        return np.broadcast_to(np.asarray(evaluate(spec['expression'], x, y, np), dtype=float), shape).copy()
    table = spec['table']
    coordinate = x if table['axis'] == 'x' else y
    return np.interp(coordinate, table['coordinates'], table['values'])
