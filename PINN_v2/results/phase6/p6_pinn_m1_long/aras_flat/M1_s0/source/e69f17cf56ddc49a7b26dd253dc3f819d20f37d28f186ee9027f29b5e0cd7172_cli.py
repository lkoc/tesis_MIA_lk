"""Single entry point: python -m pinn_v2 <command>.

Commands: help, cases, show, validate, train, fem, search. Every command reads the same case
(an id from the catalog or a JSON file), so a PINN run and a FEM run always share the data.
Run-settings files (JSON) hold the same options as the flags; a flag overrides the file.
"""
import argparse
import json
import sys
from pathlib import Path

HELP = {
    'overview': """PINN_v2 - 2D steady heat conduction in buried cables, PINN versus FEM.

  python -m pinn_v2 cases                      list the available cases
  python -m pinn_v2 show xlpe_single           summary of one case
  python -m pinn_v2 validate my_case.json      check a case file
  python -m pinn_v2 train --case xlpe_single --model M1
  python -m pinn_v2 fem --case xlpe_single --levels 0 1 2
  python -m pinn_v2 search --config configs/search_example.json
  python -m pinn_v2 help <topic>               topics: cases models train search fem files

PINN and FEM read the same case, so a comparison always uses the same physics.
FEM and paper values are never training labels; they are used only to evaluate.""",
    'cases': """A case is one JSON file (schema pinn-v2-case-1): geometry, layers, k, current,
boundary conditions per side, contact, provenance and a data status for every group.
Use an id from 'cases' or a path:  --case xlpe_single   or   --definition my_case.json
To make a new case, copy pinn_v2/cases/v2/kim_v5_summer.json, edit it and run 'validate'.
Data status: publicado, extraido, inferido, supuesto, faltante (a missing group rejects the case).""",
    'models': """--model M0    one network for the whole domain, strong form
--model M1    one network per region, strong form, interface conditions in the loss
--model M2    one network per region, mixed form (T, qx, qy)
--model CUSP | DCSNN | FBPINN   adapted variants
Add +rba or +sa for attention weights (for example M1+rba); these use Adam only.""",
    'train': """python -m pinn_v2 train --case ID --model M1 --adam-steps 6000 --lbfgs-steps 100
  --lr 3e-3 --width 32 --depth 3 --activation tanh --seed 0 --sampling-seed 11
  --weights dirichlet=100 interface_flux=10 --points n_interior=512 n_layer=256
  --reference fem_l2.npz   (optional, evaluation only, never in the loss)
  --config run.json        (same options in a file; flags win)
  --output DIR             (must not exist; default results/runs/<case>_<model>)
Writes result.json, evaluation.npz, collocation.npz, model.pt and manifest.json.""",
    'search': """python -m pinn_v2 search --config configs/search_example.json [--trials 8] [--strategy random|grid]
The file gives the case, a base setting and a search space (lists of values for model, width,
depth, lr, activation, adam_steps, lbfgs_steps, weights.<term>, points.<name>).
Trials are ranked by a held-out physics loss on new collocation points with fixed weights.
No FEM or exact field takes part in the ranking.""",
    'fem': """python -m pinn_v2 fem --case ID --levels 0 1 2 [--mode fixed|dc_temperature] [--output DIR]
Runs FEniCSx inside WSL2 on exactly the same case. Needs the WSL environment of the docs
(variables PINN_V2_WSL_DISTRO and PINN_V2_FEM_PYTHON can change it). The npz files it writes
can be passed to 'train --reference' to evaluate a PINN against them.""",
    'files': """cases/data       catalog of cases (inherited format, one JSON each)
cases/v2         native v2 case definitions
results          every run writes its own folder; nothing is overwritten
docs             full documentation (python docs/build_docs.py builds docs/_build/html)""",
}


def resolve_definition(name):
    """CaseDefinition from a catalog id or a JSON file (v2 or inherited format)."""
    from pinn_v2.cases.definition import CaseDefinition, load_all
    path = Path(name)
    if path.suffix == '.json' and path.exists():
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if 'schema_version' in data:
            return CaseDefinition.from_dict(data)
        return CaseDefinition.from_legacy(data, 'user_file', f'User file {path.name}')
    found = load_all()
    if name not in found:
        raise SystemExit(f'Unknown case {name!r}. Run: python -m pinn_v2 cases')
    return found[name]


def read_pairs(items, allowed=None, kind='value'):
    """['a=1', 'b=2'] -> {'a': 1.0, 'b': 2.0}"""
    result = {}
    for item in items or []:
        key, sep, value = item.partition('=')
        if not sep or (allowed is not None and key not in allowed):
            raise SystemExit(f'Bad {kind} {item!r}; use name=number with name in {sorted(allowed) if allowed else "any"}')
        result[key] = float(value) if allowed is None or kind != 'points' else int(value)
    return result


def cmd_help(args):
    topic = args.topic or 'overview'
    if topic not in HELP:
        raise SystemExit(f'No help topic {topic!r}. Topics: {", ".join(HELP)}')
    print(HELP[topic])


def cmd_cases(args):
    from pinn_v2.cases.definition import load_all
    for name, case in load_all().items():
        print(f"{name:26s} {case.physical.get('kind', '?'):6s} {case.title}")


def cmd_show(args):
    case = resolve_definition(args.case)
    p = case.physical
    print(f'id          {case.id}\ntitle       {case.title}\nkind        {p.get("kind")}\nsha256      {case.sha256}')
    print(f'bounds      {p.get("bounds")}\nT0 (C)      {p.get("T0")}\ncables      {len(p.get("cables", []))}')
    print(f'boundaries  {json.dumps(case.boundary_conditions) if case.boundary_conditions else "legacy: Dirichlet T0 on the whole outer boundary"}')
    print('data status')
    for group, entry in case.data_status.items():
        print(f'  {group:20s} {entry["status"]:10s} source={entry["source"]}')
    for item in case.inconsistencies:
        print(f'inconsistency {item["field"]}: {item["resolution"]}')


def cmd_validate(args):
    try:
        case = resolve_definition(args.file)
    except SystemExit:
        raise
    except Exception as exc:
        print(f'INVALID: {exc}')
        return 1
    print(f'OK {case.id} sha256={case.sha256}')
    return 0


def settings_from(args, keys):
    """Defaults < config file < flags."""
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if getattr(args, 'config', None) else {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None and value != []:
            config[key] = value
    return config


def case_source(config):
    if config.get('definition'):
        return Path(config['definition'])
    if config.get('case'):
        return config['case']
    raise SystemExit('Give --case ID or --definition FILE (or put them in the config file)')


def cmd_train(args):
    from pinn_v2.experiments.pinn_baselines import split_spec
    from pinn_v2.solvers.pinn import trainer
    from pinn_v2.solvers.pinn.objective import TERMS
    from pinn_v2.solvers.pinn.problem import load_problem
    keys = ('case', 'definition', 'model', 'adam_steps', 'lbfgs_steps', 'lr', 'width', 'depth', 'activation', 'seed', 'sampling_seed', 'output', 'reference')
    config = settings_from(args, keys)
    config['weights'] = {**config.get('weights', {}), **read_pairs(args.weights, TERMS, 'weights')}
    config['points'] = {**config.get('points', {}), **read_pairs(args.points, ('n_interior', 'n_layer', 'n_face', 'n_boundary'), 'points')}
    source = case_source(config)
    problem = load_problem(source)
    spec = config.get('model', 'M1')
    architecture, attention = split_spec(spec)
    output = Path(config.get('output') or Path('results', 'runs', f'{problem.name}_{spec.replace("+", "_")}'))
    reference = reference_label = None
    if config.get('reference'):
        from pinn_v2.evaluation.reference import load_reference_npz
        reference, reference_label = load_reference_npz(config['reference'], problem), f'{config["reference"]}, evaluation only'
    result = trainer.run(source, architecture, output, adam_steps=config.get('adam_steps', 3000), lbfgs_steps=0 if attention else config.get('lbfgs_steps', 0),
        lr=config.get('lr', 2e-3), width=config.get('width', 32), depth=config.get('depth', 3), activation=config.get('activation', 'tanh'),
        weights=config['weights'], points=config['points'], init_seed=config.get('seed', 0), sampling_seed=config.get('sampling_seed', 11),
        log_every=config.get('log_every', 200), reference=reference, reference_label=reference_label or 'supplied', attention=attention,
        command=[sys.executable, '-m', 'pinn_v2', 'train', *sys.argv[2:]])
    evaluation = result.get('evaluation', {})
    print(f"{result['status']}  loss={result.get('final_loss')}  time={result['elapsed_s']:.0f}s  acceptance={evaluation.get('acceptance', {}).get('state')}")
    print(f'record: {output}')
    return 0 if result['status'] == 'completed' else 1


def cmd_fem(args):
    from pinn_v2.solvers.fem import adapter
    from pinn_v2.cases.definition import CaseDefinition
    config = settings_from(args, ('case', 'definition', 'levels', 'mode', 'output'))
    levels = config.get('levels', [0, 1, 2])
    name = config.get('definition') or config.get('case')
    if not name:
        raise SystemExit('Give --case ID or --definition FILE')
    path = Path(name)
    v2 = Path(__file__).with_name('cases').joinpath('v2', f'{name}.json')
    output = Path(config.get('output') or Path('results', 'runs', f'fem_{path.stem}'))
    mode = config.get('mode', 'fixed')
    if v2.exists():
        records = adapter.run_cable_fem(output, levels, definition=v2, mode=mode)
    elif path.suffix == '.json' and path.exists():
        schema = 'schema_version' in json.loads(path.read_text(encoding='utf-8-sig'))
        records = adapter.run_cable_fem(output, levels, mode=mode, **({'definition': path} if schema else {'case_file': path}))
    else:
        records = adapter.run_cable_fem(output, levels, case=name, mode=mode)
    for r in records:
        print(f"level {r['level']}  dofs {r['ndofs']}  Tmax {r['Tmax_C']:.4f} C")
    print(f'record: {output}')
    return 0


def cmd_search(args):
    from pinn_v2.experiments.hparam_search import run_search
    config = settings_from(args, ('case', 'definition', 'output', 'strategy', 'seed'))
    if args.trials is not None:
        config['trials'] = args.trials
    if 'space' not in config:
        raise SystemExit('The search needs a "space" in the config file')
    source = case_source(config)
    name = Path(str(source)).stem
    output = Path(config.get('output') or Path('results', 'runs', f'search_{name}'))
    summary = run_search(source, config['space'], output, n_trials=config.get('trials', 8), strategy=config.get('strategy', 'random'),
        seed=config.get('seed', 0), base=config.get('base', {}))
    print(f"best trial: {summary['best_trial']}  (see {output.joinpath('report.md')})")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m pinn_v2', description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('help', help='built-in help'); p.add_argument('topic', nargs='?'); p.set_defaults(run=cmd_help)
    p = sub.add_parser('cases', help='list cases'); p.set_defaults(run=cmd_cases)
    p = sub.add_parser('show', help='summarise a case'); p.add_argument('case'); p.set_defaults(run=cmd_show)
    p = sub.add_parser('validate', help='check a case file'); p.add_argument('file'); p.set_defaults(run=cmd_validate)

    def common(p):
        p.add_argument('--case'); p.add_argument('--definition'); p.add_argument('--config'); p.add_argument('--output')

    p = sub.add_parser('train', help='train one PINN'); common(p); p.set_defaults(run=cmd_train)
    p.add_argument('--model'); p.add_argument('--adam-steps', dest='adam_steps', type=int); p.add_argument('--lbfgs-steps', dest='lbfgs_steps', type=int)
    p.add_argument('--lr', type=float); p.add_argument('--width', type=int); p.add_argument('--depth', type=int); p.add_argument('--activation')
    p.add_argument('--seed', type=int); p.add_argument('--sampling-seed', dest='sampling_seed', type=int); p.add_argument('--reference')
    p.add_argument('--weights', nargs='*'); p.add_argument('--points', nargs='*')
    p = sub.add_parser('fem', help='FEniCSx reference in WSL2'); common(p); p.set_defaults(run=cmd_fem)
    p.add_argument('--levels', nargs='*', type=int); p.add_argument('--mode', choices=['fixed', 'dc_temperature'])
    p = sub.add_parser('search', help='hyperparameter search'); common(p); p.set_defaults(run=cmd_search)
    p.add_argument('--trials', type=int); p.add_argument('--strategy', choices=['random', 'grid']); p.add_argument('--seed', type=int)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'run', None):
        print(HELP['overview'])
        return 0
    return args.run(args) or 0


if __name__ == '__main__':
    sys.exit(main())
