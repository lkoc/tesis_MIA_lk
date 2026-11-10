"""The only problem repository: pinnV3/problems/<id>/problem.json."""
from pathlib import Path

from pinn_v3.problem.build import load_problem as build_from_json

ROOT = Path(__file__).resolve().parents[2] / 'problems'


def problem_path(identifier, root=ROOT):
    path = Path(identifier)
    if path.is_file():
        return path
    return Path(root) / str(identifier) / 'problem.json'


def load_problem(identifier, root=ROOT):
    path = problem_path(identifier, root)
    if not path.is_file():
        raise FileNotFoundError(f'Problem JSON not found: {path}')
    return build_from_json(path)


def load_all(root=ROOT):
    result = {}
    for path in sorted(Path(root).glob('*/problem.json')):
        problem = load_problem(path, root)
        if problem.id in result:
            raise ValueError(f'Duplicate problem id: {problem.id}')
        result[problem.id] = problem
    if not result:
        raise ValueError(f'Empty problem repository: {root}')
    return result
