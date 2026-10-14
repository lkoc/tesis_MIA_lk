"""Temperature fields on regular grids, the common language of the front-end.

PINN and FEM results are both resampled on the same two windows (the whole domain and a close-up of
the cables), so any two solutions of one case can be subtracted point by point. A grid is only a
picture of a finished solution: nothing here feeds a training loss.
"""
import numpy as np

FULL_N = 320
ZOOM_N = 300


def windows(case):
    """Plot windows [x0, x1, y0, y1]: the whole domain and a close-up around the cable group."""
    x0, x1, y0, y1 = case['bounds']
    full = [x0, x1, y0, y1]
    cables = case.get('cables') or []
    if case.get('kind') != 'cable' or not cables:
        return dict(full=full, zoom=full)
    r = case['radius']
    pad = max(2.5 * r, 0.005 * (x1 - x0))
    xs = [c[0] for c in cables]
    ys = [c[1] for c in cables]
    zoom = [max(x0, min(xs) - r - pad), min(x1, max(xs) + r + pad), max(y0, min(ys) - r - pad), min(y1, max(ys) + r + pad)]
    return dict(full=full, zoom=zoom)


def grid_axes(window, n):
    """Cell-centred axes; the longer side gets n cells."""
    x0, x1, y0, y1 = window
    w, h = x1 - x0, y1 - y0
    if w >= h:
        nx, ny = n, max(24, int(round(n * h / w)))
    else:
        ny, nx = n, max(24, int(round(n * w / h)))
    x = x0 + (np.arange(nx) + 0.5) * (w / nx)
    y = y0 + (np.arange(ny) + 0.5) * (h / ny)
    return x, y


def label_grid(domain, X, Y):
    """Region id of every grid point (-1 outside the domain). Cable layers override the soil."""
    pts = np.c_[X.ravel(), Y.ravel()]
    labels = np.full(len(pts), -1, dtype=np.int16)
    for region in domain.regions:
        labels[domain.contains(region, pts)] = region.id
    return labels.reshape(X.shape)


def label_problem(problem, X, Y):
    """Region labels for verification problems without a cable FullDomain."""
    points = np.c_[X.ravel(), Y.ravel()]
    labels = np.full(len(points), -1, dtype=np.int16)
    case = problem.case
    x0, x1, y0, y1 = problem.bounds
    inside_box = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                  (points[:, 1] >= y0) & (points[:, 1] <= y1))
    if case['kind'] == 'annulus':
        radius = np.linalg.norm(points, axis=1)
        labels[inside_box & (radius >= case['ri']) & (radius <= case['ro'])] = 0
    else:
        for region in problem.regions:
            coordinate = points[:, region.axis]
            mask = (coordinate > region.lower) & (coordinate <= region.upper) & inside_box
            if region.id == 0:
                mask |= (coordinate == region.lower) & inside_box
            labels[mask] = region.id
    return labels.reshape(X.shape)


def evaluate_pinn(model, labels, X, Y, chunk=20000):
    import torch
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    pts = np.c_[X.ravel(), Y.ravel()]
    lab = labels.ravel().astype(np.int64)
    out = np.full(len(pts), np.nan)
    inside = np.flatnonzero(lab >= 0)
    with torch.no_grad():
        for start in range(0, len(inside), chunk):
            idx = inside[start:start + chunk]
            xy = torch.as_tensor(pts[idx], dtype=dtype, device=device)
            region_ids = torch.as_tensor(lab[idx], dtype=torch.long, device=device)
            out[idx] = model.temperature(xy, region_ids).detach().cpu().numpy().ravel()
    return out.reshape(X.shape)


def build_pinn(model, domain, case, full_n=None, zoom_n=None):
    """Both windows for a trained PINN model."""
    pieces = {}
    for name, window in windows(case).items():
        x, y = grid_axes(window, (full_n or FULL_N) if name == 'full' else (zoom_n or ZOOM_N))
        X, Y = np.meshgrid(x, y)
        L = label_grid(domain, X, Y)
        pieces[name] = dict(x=x, y=y, T=evaluate_pinn(model, L, X, Y), L=L, window=window)
    return pieces


def build_problem_pinn(model, problem, full_n=None, zoom_n=None):
    """PINN grids for MMS and annulus problems that do not have a FullDomain."""
    if hasattr(problem, 'domain'):
        return build_pinn(model, problem.domain, problem.case, full_n, zoom_n)
    case = dict(problem.case)
    case.setdefault('bounds', list(problem.bounds))
    pieces = {}
    for name, window in windows(case).items():
        x, y = grid_axes(window, (full_n or FULL_N) if name == 'full' else (zoom_n or ZOOM_N))
        X, Y = np.meshgrid(x, y)
        labels = label_problem(problem, X, Y)
        pieces[name] = dict(x=x, y=y, T=evaluate_pinn(model, labels, X, Y), L=labels, window=window)
    return pieces


def build_fem(npz_path, domain, case):
    """Both windows from the P2 degrees of freedom of a FEM result (linear interpolation on their Delaunay mesh)."""
    from scipy.interpolate import LinearNDInterpolator
    data = np.load(npz_path)
    xy = data['dof_xy']
    interp = LinearNDInterpolator(xy[:, :2], np.asarray(data['dof_T'], dtype=float).ravel())
    pieces = {}
    for name, window in windows(case).items():
        x, y = grid_axes(window, FULL_N if name == 'full' else ZOOM_N)
        X, Y = np.meshgrid(x, y)
        L = label_grid(domain, X, Y)
        T = np.asarray(interp(X, Y), dtype=float)
        T[L < 0] = np.nan
        pieces[name] = dict(x=x, y=y, T=T, L=L, window=window)
    return pieces


def build_verification_fem(npz_path, problem, full_n=None, zoom_n=None):
    """Resample exact-field FEM verification records on the front-end's regular grids."""
    from scipy.interpolate import LinearNDInterpolator
    data = np.load(npz_path)
    interp = LinearNDInterpolator(np.asarray(data['xy'], dtype=float), np.asarray(data['T'], dtype=float).ravel())
    case = dict(problem.case)
    case.setdefault('bounds', list(problem.bounds))
    pieces = {}
    for name, window in windows(case).items():
        x, y = grid_axes(window, (full_n or FULL_N) if name == 'full' else (zoom_n or ZOOM_N))
        X, Y = np.meshgrid(x, y)
        labels = label_problem(problem, X, Y)
        T = np.asarray(interp(X, Y), dtype=float)
        T[labels < 0] = np.nan
        pieces[name] = dict(x=x, y=y, T=T, L=labels, window=window)
    return pieces


def save(path, pieces, names, meta):
    import json
    flat = {}
    for view, p in pieces.items():
        for key in ('x', 'y', 'T', 'L'):
            flat[f'{view}_{key}'] = p[key]
        flat[f'{view}_window'] = np.asarray(p['window'], dtype=float)
    flat['names'] = np.asarray(names)
    flat['meta'] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **flat)


def load(path):
    import json
    data = np.load(path, allow_pickle=False)
    pieces = {}
    for view in ('full', 'zoom'):
        pieces[view] = dict(x=data[f'{view}_x'], y=data[f'{view}_y'], T=data[f'{view}_T'], L=data[f'{view}_L'], window=list(data[f'{view}_window']))
    return pieces, [str(n) for n in data['names']], json.loads(str(data['meta']))

def load_pinn_model(run_dir, problem):
    """Rebuild a trained network from result.json (config) and model.pt of a PINN run folder."""
    import json
    from pathlib import Path
    import torch
    from pinn_v2.solvers.pinn.networks import build_model
    cfg = json.loads(Path(run_dir).joinpath('result.json').read_text(encoding='utf-8'))['config']
    if cfg['architecture'] == 'DeepXDE':
        from pinn_v2.solvers.pinn.deepxde_solver import load_model
        return load_model(run_dir, problem)
    model = build_model(problem, cfg['architecture'], cfg['width'], cfg['depth'], cfg['activation']).double()
    model.load_state_dict(torch.load(Path(run_dir).joinpath('model.pt'), map_location='cpu'))
    model.eval()
    return model