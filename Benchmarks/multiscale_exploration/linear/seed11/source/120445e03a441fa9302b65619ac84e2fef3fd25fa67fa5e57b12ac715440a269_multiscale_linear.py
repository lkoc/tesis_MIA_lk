"""Fixed-feature shared-trace heat solver, exploratory and fixed-source only."""
from pathlib import Path
import argparse, hashlib, json, math, platform, sys, time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmarks.multiscale_trace import TracePINN
from Benchmarks.full_pinn import PhysicsLoss, evaluate
from Benchmarks.full_domain import FullDomain
from Benchmarks.full_physics import heat_residual
from pinn_cables.pinn.pde import gradients


def residual_vector(objective):
    model = objective.model
    d = objective.domain
    if d.mode != 'fixed':
        raise ValueError('Linear solve requires prescribed sources')
    vectors = []
    balance = {r.id: next(model.parameters()).new_zeros(()) for r in d.regions}
    _, powers = objective.electrical_fields()
    for r in d.regions:
        xy = objective.interior[r.id]
        T = model.field(r, xy)[0]
        residual = heat_residual(T, xy, d.conductivity(r, xy, torch), objective.static[r.id])
        vectors.append(residual.flatten()/d.scales(r)['divergence']/math.sqrt(len(xy)))
        if r.id in powers:
            balance[r.id] = balance[r.id]-powers[r.id]
    for interface, xy, normal, length in objective.transmission:
        for rid, sign in [(interface.left, 1), (interface.right, -1)]:
            r = d.regions[rid]
            T = model.field(r, xy)[0]
            q = -d.conductivity(r, xy, torch)*gradients(T, xy)[:, :2]
            balance[rid] = balance[rid]+sign*length*(q*normal).sum(1).mean()
    for r, xy, normal, length in objective.outer:
        T = model.field(r, xy)[0]
        q = -d.conductivity(r, xy, torch)*gradients(T, xy)[:, :2]
        balance[r.id] = balance[r.id]+length*(q*normal).sum(1).mean()
    vectors.append(torch.stack(list(balance.values()))*math.sqrt(10)/d.case['power'])
    return torch.cat(vectors).detach().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source = ROOT/'Benchmarks/explicit_study/C/lr_0.0005/xlpe_single'/f'seed{args.seed}/configuration.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    domain = FullDomain(config['physics']['case'], config['physics']['source_mode'])
    model = TracePINN(domain, width=32, depth=3)
    for p in model.parameters():
        p.requires_grad_(False)
    coefficients = [model.trace_T, model.trace_q]
    for net in model.networks:
        coefficients.extend([net[-1].weight, net[-1].bias])
    for p in coefficients:
        p.requires_grad_(True)
    size = sum(p.numel() for p in coefficients)
    def assign(values):
        offset = 0
        with torch.no_grad():
            for p in coefficients:
                p.copy_(torch.as_tensor(values[offset:offset+p.numel()]).reshape(p.shape))
                offset += p.numel()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    archive = output/'source'
    archive.mkdir()
    hashes = {}
    sources = ['multiscale_linear.py','multiscale_trace.py','full_pinn.py','full_domain.py',
               'full_physics.py','full_sampling.py','cases.py','expressions.py','skin_effect.py']
    paths = [ROOT/'Benchmarks'/name for name in sources]
    paths += [ROOT/'pinn_cables/pinn/pde.py',ROOT/'docs/INVESTIGACION_PINN_MULTIESCALA_2026-09-15.md']
    for path in paths:
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        hashes[str(path.relative_to(ROOT))] = sha
        (archive/(sha+'_'+path.name)).write_bytes(content)
    config.update(variant='trace_fixed_features', solver='column-scaled SVD', rcond=1e-12,
                  adjusted_coefficients=size, hidden_parameters_fixed=True,
                  baseline_configuration=str(source), experimental=True)
    (output/'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    objective = PhysicsLoss(model, config['n'], config['n_layer'], config['n_interface'],
                            config['sampling_seed'], config['sampling'])
    np.savez_compressed(output/f'collocation_seed{args.seed}.npz',
                        **{r.name:objective.interior[r.id].detach().numpy() for r in domain.regions})
    start = time.perf_counter()
    assign(np.zeros(size))
    constant = residual_vector(objective)
    matrix = np.empty((len(constant), size))
    for j in range(size):
        unit = np.zeros(size)
        unit[j] = 1
        assign(unit)
        matrix[:, j] = residual_vector(objective)-constant
        if j % 30 == 0:
            print(json.dumps(dict(column=j,total=size,seconds=time.perf_counter()-start)), flush=True)
    build_seconds = time.perf_counter()-start
    probe = np.random.default_rng(777).normal(0, .001, size)
    assign(probe)
    actual = residual_vector(objective)
    linearity = float(np.linalg.norm(actual-(matrix@probe+constant))/max(np.linalg.norm(actual),1))
    assert linearity < 1e-8, linearity
    physical_loss = float(objective()[0].detach())
    equivalence = abs(physical_loss-float(actual@actual))/max(physical_loss,1.)
    assert equivalence < 1e-8, equivalence
    scales = np.linalg.norm(matrix, axis=0)
    assert np.all(scales > 0)
    solve_start = time.perf_counter()
    value, _, rank, singular = np.linalg.lstsq(matrix/scales, -constant, rcond=1e-12)
    solve_seconds = time.perf_counter()-solve_start
    value /= scales
    assign(value)
    final_vector = residual_vector(objective)
    final_loss, parts, regional = objective()
    training_seconds = time.perf_counter()-start
    np.savez_compressed(output/'linear_system.npz', A=matrix, b=-constant, coefficients=value,
                        column_norms=scales, singular_values=singular)
    torch.save(dict(state_dict=model.state_dict(), configuration=config), output/f'pinn_seed{args.seed}.pt')
    meta = dict(configuration=config, source_sha256=hashes, seed=args.seed,
                parameters=size, parameters_total=sum(p.numel() for p in model.parameters()),
                training_seconds=training_seconds, matrix_build_seconds=build_seconds,
                solve_seconds=solve_seconds, matrix_shape=list(matrix.shape), numerical_rank=int(rank),
                linearity_relative_error=linearity, physical_loss_equivalence_relative_error=equivalence,
                final_loss=float(final_loss.detach()), final_vector_squared_norm=float(final_vector@final_vector),
                final_parts={k:float(v.detach()) for k,v in parts.items()},
                final_region_pde={k:float(v.detach()) for k,v in regional.items()},
                fem_labels_used=0, torch=torch.__version__, numpy=np.__version__,
                python=sys.version, platform=platform.platform(), device='cpu', dtype='float64')
    (output/f'training_seed{args.seed}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    report = evaluate(model, output, args.seed, config['reference'])
    print(json.dumps(dict(accepted=report['accepted'], Tmax_C=report['Tmax_C'], seconds=training_seconds)), flush=True)


if __name__ == '__main__':
    main()
