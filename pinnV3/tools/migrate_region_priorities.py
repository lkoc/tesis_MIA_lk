"""Migrate problem documents to explicit priorities and disjoint cable layers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / 'problems'


def migrate(path):
    document = json.loads(path.read_text(encoding='utf-8'))
    regions = document['regions']
    for index, region in enumerate(regions):
        region['priority'] = index
    by_cable = {}
    for region in regions:
        name = region['id']
        if not name.startswith('cable_') or '_layer_' not in name:
            continue
        cable, layer_text = name.split('_layer_', 1)
        by_cable.setdefault(cable, []).append((int(layer_text), region))
    for layers in by_cable.values():
        layers.sort(key=lambda item: item[0])
        for layer_index, region in layers[1:]:
            shape = region['shape']
            if shape['type'] != 'disk':
                continue
            previous = next(item[1] for item in layers if item[0] == layer_index - 1)
            previous_shape = previous['shape']
            inner = previous_shape.get('radius_m', previous_shape.get('outer_radius_m'))
            if inner is None:
                raise ValueError(f'Cannot determine inner radius for {region["id"]}')
            region['shape'] = {
                'type': 'annulus',
                'center': shape['center'],
                'inner_radius_m': inner,
                'outer_radius_m': shape['radius_m'],
            }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    paths = sorted(PROBLEMS.glob('*/problem.json'))
    for path in paths:
        migrate(path)
    print(f'migrated {len(paths)} problem documents')


if __name__ == '__main__':
    main()
