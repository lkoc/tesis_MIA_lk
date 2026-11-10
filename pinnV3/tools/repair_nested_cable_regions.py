"""Make converted cable layers disjoint: core disks plus annular shells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / 'problems'


def main():
    repaired = []
    for path in sorted(PROBLEMS.glob('*/problem.json')):
        doc = json.loads(path.read_text(encoding='utf-8'))
        if doc.get('family') != 'cable':
            continue
        metadata = doc.get('metadata', {})
        conductors = metadata.get('conductors', {})
        centers = conductors.get('centers', [])
        layers = conductors.get('layers', [])
        if not centers or not layers:
            continue
        layer_by_id = {region['id']: region for region in doc['regions']}
        changed = False
        for cable_index, center in enumerate(centers):
            for layer_index, layer in enumerate(layers):
                region_id = f'cable_{cable_index}_layer_{layer_index}'
                region = layer_by_id.get(region_id)
                if region is None:
                    continue
                inner, outer, _ = layer['inner_radius_m'], layer['outer_radius_m'], layer['k_W_mK']
                if layer_index == 0:
                    shape = {'type': 'disk', 'center': center, 'radius_m': outer}
                    region['kind'] = 'conductor'
                else:
                    shape = {'type': 'annulus', 'center': center,
                             'inner_radius_m': inner, 'outer_radius_m': outer}
                    region['kind'] = 'material'
                if region['shape'] != shape:
                    region['shape'] = shape
                    changed = True
        if changed:
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            repaired.append(path.parent.name)
    print(json.dumps({'repaired': repaired, 'count': len(repaired)}))


if __name__ == '__main__':
    main()
