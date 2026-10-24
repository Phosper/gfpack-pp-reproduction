"""Render fixed examples from saved actions only; never calls the optimizer."""
import argparse
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--round7', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--portfolio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from postprocess_metrics import geometry
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#f8fafb')
    for row, index in enumerate((46, 45)):
        case = read(a.frozen/'cases'/('sample_%03d.json' % index))
        old = read(a.round7/'jobs'/('sample_%03d_pred_repair_then_gap.json' % index))
        new = read(a.results/'jobs'/('sample_%03d_pred.json' % index))
        final = read(a.portfolio/'jobs'/('sample_%03d_pred.json' % index))
        variants = [(old['stages']['float32']['actions'], 'Previous repair + gap'),
                    (new['stages']['new_repair_compact']['actions'], 'New from raw'),
                    (final['portfolio']['actions'], 'Final two-start result')]
        metrics = [geometry(case['polygons'], case['polygon_ids'], actions) for actions, _ in variants]
        width = max(g['span_x'] for g in metrics)
        for col, ((actions, title), g) in enumerate(zip(variants, metrics)):
            ax = axes[row, col]
            shapes = []
            offset = g['bounds_xyxy'][0]
            for i, (pid, action) in enumerate(zip(case['polygon_ids'], actions)):
                t = math.atan2(action[3], action[2])
                c, s = math.cos(t), math.sin(t)
                poly = Polygon([(x*c-y*s+action[0]-offset, x*s+y*c+action[1])
                                for x, y in case['polygons'][pid]])
                shapes.append(poly)
                color = plt.cm.terrain(.15+.65*(i*.61803398875 % 1))
                ax.fill(*poly.exterior.xy, facecolor=color, edgecolor='#365467', alpha=.65, linewidth=.55)
            for i, first in enumerate(shapes):
                for second in shapes[i+1:]:
                    intersection = first.intersection(second)
                    parts = [intersection] if intersection.geom_type == 'Polygon' else getattr(intersection, 'geoms', [])
                    for part in parts:
                        if part.geom_type == 'Polygon' and part.area > 0:
                            ax.fill(*part.exterior.xy, color='#d32f2f', alpha=.9)
            strict = g['strict_strip_valid']
            status = 'strict geometry passed' if strict else 'overlap remains'
            ax.set_title('%s\n%.2f%%  |  %s' % (title, g['strip_utilization']*100, status),
                         color='#14695d' if strict else '#a84925', fontsize=11, pad=12)
            ax.text(.01, -.18, 'sample %d   span %.1f   max overlap %.5g' %
                    (index, g['span_x'], g['max_pairwise_area']), transform=ax.transAxes,
                    fontsize=8, color='#425563')
            ax.set_xlim(-25, width+25)
            ax.set_ylim(-20, 1220)
            ax.set_aspect('equal')
            ax.axhline(0, color='#7e8b93', linestyle='--', linewidth=.6)
            ax.axhline(1200, color='#7e8b93', linestyle='--', linewidth=.6)
            ax.tick_params(labelsize=8, colors='#607581')
            for spine in ax.spines.values():
                spine.set_color('#c5cfd5')
    fig.suptitle('Stored float32 layouts | height 1200 | same x-scale within each row', fontsize=13, color='#243d4a')
    fig.text(.5, .015, 'Display aligned at the left edge; stored actions unchanged. Red marks intersections. Fixed development examples, not a generalization result.',
             ha='center', fontsize=8, color='#526774')
    fig.tight_layout(rect=(0, .05, 1, .96), h_pad=3)
    fig.savefig(a.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
