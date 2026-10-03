"""Pictures for the front-end: geometry, temperature fields, error maps, profiles.

Everything is drawn from grids made by grids.py. Matplotlib figures are built without pyplot and
serialised under one lock, because the HTTP server answers from several threads.
"""
import io
import threading

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

LOCK = threading.Lock()
DEG = '\u00b0C'
CMAPS = ('inferno', 'viridis', 'magma', 'plasma', 'turbo', 'coolwarm', 'cividis')
ERROR_MODES = dict(signed='Error con signo (candidata menos referencia)', abs='Error absoluto', rel='Error relativo al salto termico (%)')
PALETTE = ('#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2')


def to_png(fig):
    buf = io.BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format='png', dpi=fig.dpi)
    return buf.getvalue()


def _interfaces(case):
    from pinn_v2.cases.catalog import interfaces
    try:
        return interfaces(case)
    except Exception:
        return []


def _aspect(window, floor=0.35, ceil=1.1):
    return max(floor, min(ceil, (window[3] - window[2]) / (window[1] - window[0])))


def draw_overlay(ax, case, window, layers=True, color='#111827', lw=0.7):
    """Cable circles (every layer), soil interfaces and the conductivity patch, clipped to the window."""
    x0, x1, y0, y1 = window
    for cx, cy in case.get('cables') or []:
        if cx < x0 - 1 or cx > x1 + 1:
            continue
        rows = case.get('layers') or [[0., case.get('radius', 0.), 0.]]
        for ri, ro, _ in (rows if layers else rows[-1:]):
            ax.add_patch(Circle((cx, cy), ro, fill=False, ec=color, lw=lw, alpha=0.85))
    if case.get('kind') == 'annulus':
        for radius in (case['ri'], case['ro']):
            ax.add_patch(Circle((0., 0.), radius, fill=False, ec=color, lw=lw, alpha=0.85))
    for cut in _interfaces(case):
        if cut['axis'] == 'y':
            ax.plot([x0, x1], [cut['position']] * 2, color=color, lw=lw, ls='--', alpha=0.75)
        else:
            ax.plot([cut['position']] * 2, [y0, y1], color=color, lw=lw, ls='--', alpha=0.75)
    patch = case.get('patch')
    if patch:
        cx, cy, w, h = patch[:4]
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec='#0ea5e9', lw=1.1, ls=':'))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _axes_style(ax, window):
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])


def _image(ax, piece, values, cmap, vmin, vmax, isotherms=False):
    x0, x1, y0, y1 = piece['window']
    img = ax.imshow(np.ma.masked_invalid(values), origin='lower', extent=(x0, x1, y0, y1), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
    if isotherms:
        try:
            ax.contour(piece['x'], piece['y'], np.ma.masked_invalid(values), levels=12, colors='white', linewidths=0.35, alpha=0.6)
        except Exception:
            pass
    return img


def _colorbar(fig, ax, img, label):
    bar = fig.colorbar(img, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7)


def geometry_png(case, view='full', width=6.4):
    """Layout of a case on its own: domain, cables with every layer, interfaces, patch."""
    from grids import windows
    window = windows(case)[view]
    with LOCK:
        fig = Figure(figsize=(width, width * _aspect(window) + 0.7), dpi=110, layout='constrained')
        ax = fig.add_subplot(111)
        b = case['bounds']
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc='#f1f5f9', ec='#334155', lw=1.2))
        shades = ('#f59e0b', '#93c5fd', '#a7f3d0', '#fca5a5', '#ddd6fe', '#fde68a')
        for cx, cy in case.get('cables') or []:
            rows = case.get('layers') or []
            for j in range(len(rows) - 1, -1, -1):
                ax.add_patch(Circle((cx, cy), rows[j][1], fc=shades[j % len(shades)], ec='#475569', lw=0.5))
        draw_overlay(ax, case, window, layers=True, color='#334155')
        _axes_style(ax, window)
        ax.set_title(f"{case.get('id', 'caso')}  ({'dominio completo' if view == 'full' else 'zoom a los cables'})", fontsize=9)
        return to_png(fig)


def field_png(pieces, case, view='full', cmap='inferno', overlay=True, isotherms=False, vmin=None, vmax=None, title='', width=7.2):
    piece = pieces[view]
    T = piece['T']
    lo = np.nanmin(T) if vmin is None else vmin
    hi = np.nanmax(T) if vmax is None else vmax
    with LOCK:
        fig = Figure(figsize=(width, width * _aspect(piece['window']) + 0.9), dpi=110, layout='constrained')
        ax = fig.add_subplot(111)
        img = _image(ax, piece, T, cmap, lo, hi, isotherms)
        if overlay:
            draw_overlay(ax, case, piece['window'], layers=view == 'zoom')
        _axes_style(ax, piece['window'])
        _colorbar(fig, ax, img, f'T ({DEG})')
        ax.set_title(title, fontsize=9)
        return to_png(fig)


def error_field(ref, cand, mode, rise):
    d = cand - ref
    if mode == 'abs':
        return np.abs(d), 'K'
    if mode == 'rel':
        return 100. * d / rise, '% del salto'
    return d, 'K'


def _scale(values, scale):
    finite = np.abs(values[np.isfinite(values)])
    if not finite.size:
        return 1.
    if isinstance(scale, (int, float)) and scale > 0:
        return float(scale)
    top = float(finite.max()) if scale == 'max' else float(np.percentile(finite, 99.5))
    return top if top > 0 else 1.


def rise_of(ref_pieces):
    T = ref_pieces['full']['T']
    return max(float(np.nanmax(T) - np.nanmin(T)), 1e-9)


def _error_panel(fig, ax, ref_piece, cand_piece, mode, top, rise, cmap, overlay, case, view, label):
    err, unit = error_field(ref_piece['T'], cand_piece['T'], mode, rise)
    if mode == 'abs':
        img = _image(ax, ref_piece, err, cmap or 'magma', 0., top)
    else:
        img = _image(ax, ref_piece, err, cmap or 'RdBu_r', -top, top)
    if overlay:
        draw_overlay(ax, case, ref_piece['window'], layers=view == 'zoom')
    _axes_style(ax, ref_piece['window'])
    _colorbar(fig, ax, img, unit)
    ax.set_title(label, fontsize=8)


def compare_png(ref, cands, case, view='full', mode='signed', scale=None, cmap='inferno', error_cmap=None, overlay=True, detail=False, width=14.):
    """Triptych (reference, candidate, error) for one candidate; a gallery of error maps for several."""
    rise = rise_of(ref['pieces'])
    rp = ref['pieces'][view]
    with LOCK:
        if detail or len(cands) == 1:
            cand = cands[0]
            cp = cand['pieces'][view]
            lo = float(min(np.nanmin(rp['T']), np.nanmin(cp['T'])))
            hi = float(max(np.nanmax(rp['T']), np.nanmax(cp['T'])))
            fig = Figure(figsize=(width, width / 3 * _aspect(rp['window'], 0.3) + 1.0), dpi=100, layout='constrained')
            axes = fig.subplots(1, 3)
            for ax, piece, label in ((axes[0], rp, 'Referencia: ' + ref['label']), (axes[1], cp, 'Candidata: ' + cand['label'])):
                img = _image(ax, piece, piece['T'], cmap, lo, hi)
                if overlay:
                    draw_overlay(ax, case, piece['window'], layers=view == 'zoom')
                _axes_style(ax, piece['window'])
                ax.set_title(label, fontsize=8)
                _colorbar(fig, ax, img, f'T ({DEG})')
            err, _ = error_field(rp['T'], cp['T'], mode, rise)
            _error_panel(fig, axes[2], rp, cp, mode, _scale(err, scale), rise, error_cmap, overlay, case, view, ERROR_MODES[mode].split(' (')[0])
            return to_png(fig)
        cands = cands[:8]
        n = len(cands)
        cols = min(3, n)
        rows = -(-n // cols)
        errs = [error_field(rp['T'], c['pieces'][view]['T'], mode, rise)[0] for c in cands]
        top = max(_scale(e, scale) for e in errs)
        fig = Figure(figsize=(width, (width / cols * _aspect(rp['window'], 0.3) + 0.9) * rows), dpi=100, layout='constrained')
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, cand in zip(axes, cands):
            _error_panel(fig, ax, rp, cand['pieces'][view], mode, top, rise, error_cmap, overlay, case, view, cand['label'])
        for ax in axes[n:]:
            ax.axis('off')
        return to_png(fig)


def profile_png(ref, cands, axis='x', at=0., mode='signed', width=9.):
    """T along a vertical (axis x) or horizontal (axis y) line, plus the error histogram of every candidate."""
    from scipy.interpolate import RegularGridInterpolator
    with LOCK:
        fig = Figure(figsize=(width, 3.6), dpi=100, layout='constrained')
        a1, a2 = fig.subplots(1, 2)
        for color, sol in zip(('#111827',) + PALETTE, [ref] + list(cands)):
            p = sol['pieces']['full']
            f = RegularGridInterpolator((p['y'], p['x']), p['T'], bounds_error=False, fill_value=np.nan)
            if axis == 'x':
                s = np.linspace(p['window'][2], p['window'][3], 600)
                pts = np.c_[s, np.full_like(s, at)]
            else:
                s = np.linspace(p['window'][0], p['window'][1], 600)
                pts = np.c_[np.full_like(s, at), s]
            a1.plot(s, f(pts), color=color, lw=1.6 if sol is ref else 1.1, ls='-' if sol is ref else '--', label=sol['label'])
        a1.set_xlabel('y (m)' if axis == 'x' else 'x (m)', fontsize=8)
        a1.set_ylabel(f'T ({DEG})', fontsize=8)
        a1.set_title(f"Perfil en {axis} = {at:g} m", fontsize=9)
        a1.tick_params(labelsize=7)
        a1.grid(alpha=0.25)
        a1.legend(fontsize=6.5)
        rise = rise_of(ref['pieces'])
        for color, cand in zip(PALETTE, cands):
            if cand['pieces']['full']['T'].shape != ref['pieces']['full']['T'].shape:
                continue
            err, _ = error_field(ref['pieces']['full']['T'], cand['pieces']['full']['T'], 'rel' if mode == 'rel' else 'signed', rise)
            err = err[np.isfinite(err)]
            if err.size:
                span = max(float(np.percentile(np.abs(err), 99.5)), 1e-9)
                a2.hist(err, bins=80, range=(-span, span), histtype='step', color=color, lw=1.2, label=cand['label'])
        a2.set_xlabel('error (% del salto)' if mode == 'rel' else 'error (K)', fontsize=8)
        a2.set_ylabel('celdas', fontsize=8)
        a2.set_yscale('log')
        a2.set_title('Distribucion del error', fontsize=9)
        a2.tick_params(labelsize=7)
        a2.grid(alpha=0.25)
        a2.legend(fontsize=6.5)
        return to_png(fig)


def metrics(ref, cand, top_regions=12):
    """Error numbers of one candidate against the reference on the full grid (and the conductor on the zoom grid)."""
    rp, cp = ref['pieces']['full'], cand['pieces']['full']
    if rp['T'].shape != cp['T'].shape:
        return dict(label=cand['label'], error='las mallas de comparacion no coinciden (dominio distinto)')
    d = cp['T'] - rp['T']
    ok = np.isfinite(d) & (rp['L'] >= 0)
    if not ok.any():
        return dict(label=cand['label'], error='no hay celdas comunes validas')
    e = d[ok]
    rise = rise_of(ref['pieces'])
    at = int(np.argmax(np.abs(e)))
    yy, xx = np.meshgrid(rp['y'], rp['x'], indexing='ij')
    out = dict(label=cand['label'], rmse_K=float(np.sqrt(np.mean(e ** 2))), mae_K=float(np.mean(np.abs(e))), max_abs_K=float(np.abs(e).max()), bias_K=float(e.mean()),
        nrmse_pct=float(100 * np.sqrt(np.mean(e ** 2)) / rise), max_at=[float(xx[ok][at]), float(yy[ok][at])], rise_K=rise, cells=int(e.size))
    zr, zc = ref['pieces']['zoom'], cand['pieces']['zoom']
    hot = [i for i, n in enumerate(ref['names']) if n.endswith('_layer_0')]
    if hot and zr['T'].shape == zc['T'].shape:
        mask = np.isin(zr['L'], hot) & np.isfinite(zr['T']) & np.isfinite(zc['T'])
        if mask.any():
            out.update(conductor_tmax_ref_C=float(zr['T'][mask].max()), conductor_tmax_cand_C=float(zc['T'][mask].max()), conductor_tmax_diff_K=float(zc['T'][mask].max() - zr['T'][mask].max()))
    rows = []
    for i, n in enumerate(ref['names']):
        m = ok & (rp['L'] == i)
        if m.sum() >= 4:
            rows.append(dict(region=n, rmse_K=float(np.sqrt(np.mean(d[m] ** 2))), max_abs_K=float(np.abs(d[m]).max()), cells=int(m.sum())))
    rows.sort(key=lambda r: -r['rmse_K'])
    out['regions'] = rows[:top_regions]
    out['regions_total'] = len(rows)
    return out