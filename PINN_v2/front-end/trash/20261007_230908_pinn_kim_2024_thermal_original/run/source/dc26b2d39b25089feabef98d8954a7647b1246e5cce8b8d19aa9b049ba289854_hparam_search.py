"""Hyperparameter search for the PINN models (random or grid).

Each trial is a normal trainer run with its own never-overwritten folder. Trials are ranked by
a held-out physics loss: the same terms and the same fixed weights for every trial, evaluated
on collocation points drawn with a seed that training never uses. No reference solution
(exact, FEM or paper) takes part in the ranking. If a reference is given, its error is
stored next to each trial for information only, and the summary says so.

The search space is a dictionary of lists. Keys: model, width, depth, lr, activation,
adam_steps, lbfgs_steps, and dotted keys weights.<term> and points.<name>.
"""
import itertools
import math
import random
import sys
from pathlib import Path

import torch

from pinn_v2.manifest.run_manifest import build_manifest, write_manifest, write_json_atomic
from pinn_v2.solvers.pinn import trainer
from pinn_v2.solvers.pinn.networks import build_model
from pinn_v2.solvers.pinn.objective import DEFAULT_WEIGHTS, Objective
from pinn_v2.solvers.pinn.problem import load_problem

HELD_OUT_SEED = 424243
TOP_KEYS = ('model', 'width', 'depth', 'lr', 'activation', 'adam_steps', 'lbfgs_steps')
BASE = dict(model='M1', width=32, depth=3, lr=3e-3, activation='tanh', adam_steps=2000, lbfgs_steps=0)


def split_model(spec):
    """'M1+rba' gives ('M1', 'rba'); a plain name has no attention."""
    name, _, attention = spec.partition('+')
    return name, (attention or None)


def candidates(space, n_trials, strategy='random', seed=0):
    """List of trial settings drawn from the search space."""
    keys = sorted(space)
    for key in keys:
        if not space[key]:
            raise ValueError(f'Empty list for {key}')
        if key not in TOP_KEYS and not key.startswith(('weights.', 'points.')):
            raise ValueError(f'Unknown search key {key!r}')
    if strategy == 'grid':
        grid = [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
        return grid[:n_trials]
    rng = random.Random(seed)
    seen, trials = set(), []
    total = math.prod(len(space[k]) for k in keys)
    while len(trials) < min(n_trials, total):
        choice = tuple(rng.choice(space[k]) for k in keys)
        if choice not in seen:
            seen.add(choice)
            trials.append(dict(zip(keys, choice)))
    return trials


def trial_kwargs(choice, base):
    """Split one flat choice into the keyword arguments of trainer.run."""
    flat = {**BASE, **base, **choice}
    weights = dict(base.get('weights', {}))
    points = dict(base.get('points', {}))
    for key, value in choice.items():
        if key.startswith('weights.'):
            weights[key[8:]] = value
        elif key.startswith('points.'):
            points[key[7:]] = value
    model, attention = split_model(flat['model'])
    kwargs = dict(width=flat['width'], depth=flat['depth'], lr=flat['lr'], activation=flat['activation'],
        adam_steps=flat['adam_steps'], lbfgs_steps=0 if attention else flat['lbfgs_steps'], weights=weights, points=points, attention=attention)
    return model, kwargs


def held_out_loss(problem, architecture, folder, config):
    """Reload the trained network and score it on fresh points with the default weights."""
    model = build_model(problem, architecture, config['width'], config['depth'], config['activation']).double()
    model.load_state_dict(torch.load(Path(folder).joinpath('model.pt')))
    objective = Objective(model, seed=HELD_OUT_SEED, weights=dict(DEFAULT_WEIGHTS), **config['points'])
    total, parts, _ = objective()
    return float(total.detach()), {k: float(v.detach()) for k, v in parts.items()}


def run_search(source, space, output, n_trials=8, strategy='random', seed=0, base=None, reference=None, reference_label='supplied'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = dict(base or {})
    problem = load_problem(source)
    rows = []
    for index, choice in enumerate(candidates(space, n_trials, strategy, seed)):
        architecture, kwargs = trial_kwargs(choice, base)
        folder = output.joinpath(f'trial_{index:03d}')
        result = trainer.run(source, architecture, folder, init_seed=base.get('init_seed', 0), sampling_seed=base.get('sampling_seed', 11),
            study=output.name, log_every=500, reference=reference, reference_label=reference_label, **kwargs)
        row = dict(trial=index, choice=choice, status=result['status'], final_loss=result.get('final_loss'), elapsed_s=result['elapsed_s'],
            held_out_loss=None, held_out_parts=None, reference_error_pct=None, error=(result.get('error') or {}).get('message'))
        if result['status'] == 'completed':
            try:
                row['held_out_loss'], row['held_out_parts'] = held_out_loss(problem, architecture, folder, result['config'])
            except Exception as exc:
                row['status'], row['error'] = 'failed', f'held-out scoring: {exc}'
            evaluation = result.get('evaluation', {})
            row['reference_error_pct'] = evaluation.get('error_rise_pct')
        rows.append(row)
        print(index, choice, row['status'], row['held_out_loss'], flush=True)
    ranked = sorted((r for r in rows if r['held_out_loss'] is not None and math.isfinite(r['held_out_loss'])), key=lambda r: r['held_out_loss'])
    summary = dict(case=problem.name, strategy=strategy, seed=seed, space=space, base=base, held_out_seed=HELD_OUT_SEED, rows=rows,
        best_trial=ranked[0]['trial'] if ranked else None, ranking='held-out physics loss with fixed default weights',
        selection_uses_reference=False, reference_label=reference_label if reference is not None else None, fem_labels_used=0)
    write_json_atomic(output.joinpath('summary.json'), summary)
    write_manifest(output, build_manifest('pinn_hparam_search', [sys.executable, '-m', 'pinn_v2.experiments.hparam_search'], seeds=dict(search=seed, held_out=HELD_OUT_SEED),
        extra=dict(case=problem.name, trials=len(rows), selection_uses_reference=False)))
    write_report(output, summary, ranked)
    return summary


def write_report(output, summary, ranked):
    lines = [f"# Hyperparameter search on {summary['case']}", '', f"Strategy {summary['strategy']}; ranking: {summary['ranking']}.",
        'The reference error column is informative only and never enters the ranking.', '',
        '| rank | trial | choice | status | held-out loss | final train loss | reference dT error % |', '|---|---|---|---|---|---|---|']
    order = {r['trial']: i + 1 for i, r in enumerate(ranked)}
    for r in sorted(summary['rows'], key=lambda r: order.get(r['trial'], 10 ** 6)):
        fmt = lambda v: '' if v is None else format(v, '.4g')
        lines.append(f"| {order.get(r['trial'], '-')} | {r['trial']} | {r['choice']} | {r['status']} | {fmt(r['held_out_loss'])} | {fmt(r['final_loss'])} | {fmt(r['reference_error_pct'])} |")
    Path(output).joinpath('report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
