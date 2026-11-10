"""Problem entry point for every PINN family: each problem is a pinn-v3-problem-1 document."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Face:
    """Interface samples. xy lies inside the left region; xy_right (if given) lies inside the right region."""
    name: str
    left: int
    right: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    axis: int = -1
    xy_right: np.ndarray = None


@dataclass(frozen=True)
class Piece:
    side: str
    region: int
    xy: np.ndarray
    normal: np.ndarray
    weight: float
    bc: dict


def load_problem(source):
    """Build a GenericProblem from a document, a canonical problem, a problem file, or a problem identifier."""
    from pinn_v3.problem.build import Problem as ProblemDefinition, build_problem
    from pinn_v3.problem.repository import load_problem as load_identifier
    from pinn_v3.solvers.pinn.generic_problem import GenericProblem

    if isinstance(source, GenericProblem):
        return source
    if isinstance(source, (dict, ProblemDefinition)):
        return GenericProblem(source)
    text = str(source)
    if isinstance(source, Path) or text.lower().endswith('.json'):
        return GenericProblem(build_problem(json.loads(Path(source).read_text(encoding='utf-8-sig'))))
    return GenericProblem(load_identifier(text))
