#!/usr/bin/env python3
"""Independently replay frozen concave-compaction evidence without the solver.

This verifier reconstructs contours with Shapely affine operations and converts
storage values with struct, independently of the runner's numpy geometry.  It
does not import compact_layout, translation_intervals, or the runner's geometry.
Only the aggregation of independently checked rows uses the shared summarize.
The public report contains metrics and counts, never complete actions/contours.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct
import sys

sys.dont_write_bytecode = True

MANIFEST_SHA = '9b0fd899a72e29027906751fe9caf4386f760eddf3d17124a5b9fe49732090c6'
SAMPLES = (0, 17, 25, 35, 45, 46, 47, 54, 56, 62, 74, 91, 99, 109, 115, 127)
FAILURES = (25, 45, 46, 47, 54, 62, 99, 115)
ROUTES = ('new_repair', 'new_repair_compact', 'saved_repair_compact')
OLD_ROUTES = ('gap_only', 'repair_only', 'repair_then_gap')
SOURCES = {'pred': 'raw_actions', 'gt': 'gt_actions'}
HEIGHT = 1200.
EPS = 1e-9
GEOMETRY_ATOL = 1e-7
SCRIPT_INPUTS = ('compact_layout.py', 'translation_intervals.py',
                 'run_compact_layouts.py', 'postprocess_metrics.py',
                 'check_collision_saved_cases.py')


class VerificationError(ValueError):
    def __init__(self, code, location='suite'):
        self.code, self.location = code, location
        super().__init__(code)


def require(condition, code, location='suite'):
    if not condition:
        raise VerificationError(code, location)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise VerificationError('nonfinite_json_constant', path.name)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def contained(root, name):
    require(isinstance(name, str), 'invalid_relative_path')
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and '..' not in rel.parts and
            '\\' not in name and ':' not in name, 'unsafe_relative_path')
    path = (root / Path(*rel.parts)).resolve()
    require(root in path.parents, 'path_outside_evidence_root')
    return path


def close(actual, expected, location, atol=1e-10, rtol=1e-11):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected),
                'dictionary_keys_disagree', location)
        for key, value in expected.items():
            close(actual[key], value, location+'/'+str(key), atol, rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected),
                'list_length_disagrees', location)
        for i, value in enumerate(expected):
            close(actual[i], value, location+'/'+str(i), atol, rtol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected,
                'value_disagrees', location)
    else:
        require(type(actual) in (int, float) and type(expected) in (int, float)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                'number_disagrees', location)


def scalar32(value):
    return struct.unpack('=f', struct.pack('=f', value))[0]


def fp32(actions):
    return [[scalar32(v) for v in a] for a in actions]


def next32(value, upward):
    """Adjacent representable float32 without numpy or solver helpers."""
    value = scalar32(value)
    if value == 0:
        return struct.unpack('=f', struct.pack('=I', 1 if upward else 0x80000001))[0]
    bits = struct.unpack('=I', struct.pack('=f', value))[0]
    bits += 1 if (upward == (value > 0)) else -1
    return struct.unpack('=f', struct.pack('=I', bits))[0]


def actions_checked(actions, n, location, stored=False, rotations=None):
    require(isinstance(actions, list) and len(actions) == n,
            'action_piece_count_changed', location)
    for i, a in enumerate(actions):
        require(isinstance(a, list) and len(a) == 4 and
                all(type(v) in (int, float) and math.isfinite(v) for v in a)
                and (a[2] != 0 or a[3] != 0), 'invalid_actions', location)
        if stored:
            require(all(v == scalar32(v) for v in a), 'not_stored_float32', location)
        if rotations is not None:
            require(a[2:] == rotations[i][2:], 'rotation_changed_after_cast', location)


def base_polygons(case, actions, location):
    from shapely.geometry import Polygon
    from shapely.affinity import rotate
    result = []
    for pid, action in zip(case['polygon_ids'], actions):
        source = Polygon(case['polygons'][pid])
        require(source.is_valid and not source.is_empty and source.area > 0,
                'invalid_source_contour', location)
        angle = math.atan2(action[3], action[2])
        result.append(rotate(source, angle, origin=(0., 0.), use_radians=True))
    return result


def placed(base, actions):
    from shapely.affinity import translate
    return [translate(p, xoff=a[0], yoff=a[1]) for p, a in zip(base, actions)]


def pair_area(a, b):
    aa, bb = a.bounds, b.bounds
    if aa[2] < bb[0] or bb[2] < aa[0] or aa[3] < bb[1] or bb[3] < aa[1]:
        return 0.
    value = float(a.intersection(b).area)
    require(math.isfinite(value) and value >= 0, 'invalid_intersection')
    return value


def geometry(case, actions, location):
    actions_checked(actions, len(case['polygon_ids']), location)
    shapes = placed(base_polygons(case, actions, location), actions)
    require(all(p.is_valid and p.area > 0 for p in shapes), 'invalid_placed_shape', location)
    bounds = [min(p.bounds[0] for p in shapes), min(p.bounds[1] for p in shapes),
              max(p.bounds[2] for p in shapes), max(p.bounds[3] for p in shapes)]
    areas = [(pair_area(a, b), i, j) for i, a in enumerate(shapes)
             for j, b in enumerate(shapes[i+1:], i+1)]
    areas = sorted((v for v in areas if v[0] > 0), reverse=True)
    maximum = max((v[0] for v in areas), default=0.)
    overlap = math.fsum(v[0] for v in areas)
    total = math.fsum(p.area for p in shapes)
    from shapely.geometry import Polygon
    original = math.fsum(Polygon(case['polygons'][pid]).area for pid in case['polygon_ids'])
    sx, sy = bounds[2]-bounds[0], bounds[3]-bounds[1]
    violation = max(0., -bounds[1], bounds[3]-HEIGHT)
    require(sx > 0 and sy > 0 and total > 0, 'invalid_aggregate_extent', location)
    return {'status': 'measured', 'piece_count': len(shapes), 'bounds_xyxy': bounds,
            'total_polygon_area': total, 'original_polygon_area': original,
            'rigid_area_absolute_error': abs(total-original),
            'max_pairwise_area': maximum, 'sum_pairwise_area': overlap,
            'pair_count_over_50': sum(v[0] > 50 for v in areas),
            'pair_count_over_1e_9': sum(v[0] > EPS for v in areas),
            'positive_area_pair_count': len(areas),
            'sum_pairwise_overlap_fraction': overlap/total,
            'span_x': sx, 'span_y': sy, 'height_violation': violation,
            'lower_y_violation': max(0., -bounds[1]),
            'upper_y_violation': max(0., bounds[3]-HEIGHT),
            'bbox_utilization': total/(sx*sy), 'strip_utilization': total/(HEIGHT*sx),
            'strict_overlap_valid': maximum <= EPS, 'eps50_overlap_valid': maximum <= 50,
            'strict_strip_valid': maximum <= EPS and violation <= EPS,
            'eps50_strip_valid': maximum <= 50 and violation <= EPS,
            'within_strip': violation <= EPS, 'strip_height': HEIGHT,
            'zero_rotation_vector_count': 0,
            'horizontal_domain': 'infinite_strip_no_x_wall',
            'area_eps': 50., 'strict_area_eps': EPS, 'boundary_eps': EPS,
            'top_pairs': [{'area': v, 'piece_indices': [i, j]} for v, i, j in areas[:3]]}


def check_geometry(actual, expected, location):
    require(isinstance(actual, dict), 'missing_geometry', location)
    for key, value in expected.items():
        if key in ('top_pairs', 'positive_area_pair_count'):
            # Affine roundoff may change only sub-tolerance positive contacts.
            # All unthresholded area totals and both threshold counts are checked.
            continue
        require(key in actual, 'missing_geometry_field', location+'/'+key)
        close(actual[key], value, location+'/'+key, GEOMETRY_ATOL)


def comparison(before, after, original, output):
    shifts = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(original, output)]
    distance = [math.hypot(x, y) for x, y in shifts]
    means = [math.fsum(v[k] for v in shifts)/len(shifts) for k in (0, 1)]
    delta = {k: after[k]-before[k] for k in ('max_pairwise_area', 'sum_pairwise_area',
             'height_violation', 'bbox_utilization', 'strip_utilization', 'span_x', 'span_y')}
    rotations = all(a[2:] == b[2:] for a, b in zip(original, output))
    worse = delta['max_pairwise_area'] > EPS or delta['sum_pairwise_area'] > EPS
    gauge = rotations and all(y == 0 for _, y in shifts) and all(
        abs(x-means[0]) <= EPS for x, _ in shifts)
    invariants = {k: math.isclose(before[k], after[k], rel_tol=1e-10, abs_tol=1e-7)
                  for k in ('max_pairwise_area', 'sum_pairwise_area', 'bbox_utilization',
                            'strip_utilization', 'height_violation', 'span_x', 'span_y',
                            'total_polygon_area')}
    return {'delta': delta, 'overlap_worse': worse,
            'overlap_improved': not worse and (delta['max_pairwise_area'] < -EPS or
                                              delta['sum_pairwise_area'] < -EPS),
            'bounds_improved': delta['height_violation'] < -EPS,
            'bounds_worse': delta['height_violation'] > EPS,
            'bbox_utilization_drop': delta['bbox_utilization'] < -1e-12,
            'strip_utilization_drop': delta['strip_utilization'] < -1e-12,
            'became_eps50_strip_valid': not before['eps50_strip_valid'] and after['eps50_strip_valid'],
            'lost_eps50_strip_validity': before['eps50_strip_valid'] and not after['eps50_strip_valid'],
            'became_strict_strip_valid': not before['strict_strip_valid'] and after['strict_strip_valid'],
            'lost_strict_strip_validity': before['strict_strip_valid'] and not after['strict_strip_valid'],
            'translation_rms': math.sqrt(math.fsum(v*v for v in distance)/len(distance)),
            'translation_max': max(distance), 'translation_mean_xy': means,
            'relative_translation_rms': math.sqrt(math.fsum(
                (x-means[0])**2+(y-means[1])**2 for x, y in shifts)/len(shifts)),
            'rotation_unchanged': rotations,
            'rotation_component_max_change': max(abs(b[k]-a[k]) for a, b in
                                                 zip(original, output) for k in (2, 3)),
            'x_gauge_only_change': gauge,
            'x_gauge_invariance': all(invariants.values()) if gauge else None,
            'x_gauge_invariance_checks': invariants if gauge else None}


class Replay:
    """Pair cache is safe: each changed piece is retested against every other."""
    def __init__(self, case, actions, location, project_boundary=False):
        self.location = location
        self.actions = [a[:] for a in actions]
        self.base = base_polygons(case, actions, location)
        self.boundary_adjustments = 0
        if project_boundary:
            for i, p in enumerate(self.base):
                lo, hi = -p.bounds[1], HEIGHT-p.bounds[3]
                lower, upper = scalar32(lo), scalar32(hi)
                if lower < lo:
                    lower = next32(lower, True)
                if upper > hi:
                    upper = next32(upper, False)
                require(lower <= upper, 'piece_cannot_fit_at_stored_rotation', location)
                y = self.actions[i][1]
                if not lower <= y <= upper:
                    self.actions[i][1] = min(upper, max(lower, y))
                    self.boundary_adjustments += 1
        self.shapes = placed(self.base, self.actions)
        n = len(actions)
        self.pairs = [[0.]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                self.pairs[i][j] = self.pairs[j][i] = pair_area(self.shapes[i], self.shapes[j])
        self.counts = Counter()

    def span(self):
        return max(p.bounds[2] for p in self.shapes)-min(p.bounds[0] for p in self.shapes)

    def strict(self):
        return max(max(row) for row in self.pairs) <= EPS and all(
            p.bounds[1] >= -EPS and p.bounds[3] <= HEIGHT+EPS for p in self.shapes)

    def move(self, record):
        from shapely.affinity import translate
        loc = self.location+'/move_'+str(self.counts['moves'])
        phase, i = record['phase'], record['piece_index']
        require(type(i) is int and 0 <= i < len(self.actions), 'invalid_moved_piece', loc)
        require(phase in ('local_repair', 'compact_left', 'compact_right'), 'unknown_move_phase', loc)
        require(record['before_xy'] == self.actions[i][:2], 'move_before_not_replay_state', loc)
        xy = record['after_xy']
        require(isinstance(xy, list) and len(xy) == 2 and all(type(v) in (int, float)
                and math.isfinite(v) and scalar32(v) == v for v in xy), 'move_not_float32', loc)
        old_span = self.span()
        close(record['span_before'], old_span, loc+'/span_before', GEOMETRY_ATOL)
        if phase != 'local_repair':
            require(self.strict(), 'compaction_started_from_infeasible_layout', loc)
            require(xy[1] == self.actions[i][1], 'nonhorizontal_compaction_move', loc)
            require((xy[0] <= self.actions[i][0] if phase == 'compact_left' else
                     xy[0] >= self.actions[i][0]), 'compaction_direction_disagrees', loc)
        self.shapes[i] = translate(self.base[i], xoff=xy[0], yoff=xy[1])
        self.actions[i][:2] = xy[:]
        for j in range(len(self.actions)):
            if j != i:
                area = pair_area(self.shapes[i], self.shapes[j])
                self.pairs[i][j] = self.pairs[j][i] = area
                self.counts['moved_piece_pair_rechecks'] += 1
        require(max(self.pairs[i]) <= EPS, 'accepted_move_collides_with_another_piece', loc)
        require(all(p.bounds[1] >= -EPS and p.bounds[3] <= HEIGHT+EPS for p in self.shapes),
                'accepted_move_outside_height', loc)
        new_span = self.span()
        close(record['span_after'], new_span, loc+'/span_after', GEOMETRY_ATOL)
        if phase != 'local_repair':
            require(self.strict(), 'accepted_compaction_not_globally_strict', loc)
            require(new_span <= old_span+EPS, 'accepted_compaction_increases_span', loc)
            self.counts['global_strict_compaction_endpoints'] += 1
        self.counts['moves'] += 1
        self.counts[phase] += 1
        self.counts['stored_noop_moves'] += int(record['before_xy'] == xy)


def brief(g):
    return {k: g[k] for k in ('max_pairwise_area', 'sum_pairwise_area', 'span_x',
            'strip_utilization', 'bbox_utilization', 'bounds_xyxy', 'height_violation',
            'strict_strip_valid', 'eps50_strip_valid')}


def stats(values):
    return {'count': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'mean': math.fsum(values)/len(values) if values else None,
            'max': max(values) if values else None}


def aggregate_brief(aggregate):
    return [{k: group[k] for k in ('path', 'source', 'baseline_group', 'output_precision',
              'planned_outputs', 'measured_outputs', 'failed_outputs', 'strict_strip_valid',
              'eps50_strip_valid', 'within_strip', 'strip_utilization',
              'strict_strip_valid_strip_utilization', 'eps50_strip_valid_strip_utilization')}
            for group in aggregate['by_path_source_group']]


def verify(args):
    frozen, oldroot, root = args.frozen.resolve(), args.round7.resolve(), args.results.resolve()
    scriptroot = Path(__file__).resolve().parent
    manifest = read(frozen/'manifest.json')
    require(sha(frozen/'manifest.json') == MANIFEST_SHA, 'frozen_manifest_changed')
    require((frozen/'manifest.sha256').read_text().split()[0] == MANIFEST_SHA, 'manifest_sidecar_changed')
    entries = manifest['cases']
    require(len(entries) == 16 and sorted(e['sample_index'] for e in entries) == list(SAMPLES),
            'frozen_sample_coverage')
    oldmeta, meta = read(oldroot/'metadata.json'), read(root/'metadata.json')
    require(oldmeta['status'] == meta['status'] == 'completed' and
            oldmeta['changed_inputs'] == meta['changed_inputs'] == [], 'incomplete_or_changed_inputs')
    for directory, metadata in ((oldroot, oldmeta), (root, meta)):
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                  if p.is_file() and p != directory/'metadata.json'}
        require(actual == set(metadata['output_sha256']), 'output_inventory_incomplete')
        for rel, digest in metadata['output_sha256'].items():
            require(sha(contained(directory, rel)) == digest, 'output_hash_disagrees', rel)
    inputs = read(root/'inputs.json')
    expected = {'scripts/'+name: sha(scriptroot/name) for name in SCRIPT_INPUTS}
    expected.update({'frozen/manifest.json': MANIFEST_SHA,
                     'round7/metadata.json': sha(oldroot/'metadata.json')})
    for e in entries:
        index = e['sample_index']
        require(e['path'] == 'cases/sample_%03d.json' % index and e['group'] ==
                ('failure_development' if index in FAILURES else 'success_regression'),
                'frozen_case_identity')
        require(sha(contained(frozen, e['path'])) == e['sha256'], 'case_hash_disagrees')
        expected['frozen/'+e['path']] = e['sha256']
    expected.update({'round7/'+rel: digest for rel, digest in oldmeta['output_sha256'].items()})
    require(inputs == expected and meta['input_count'] == len(expected), 'input_inventory_disagrees')
    plan, summary = read(root/'plan.json'), read(root/'summary.json')
    require(plan['samples'] == [e['sample_index'] for e in entries] and
            plan['sources'] == SOURCES and plan['routes'] == list(ROUTES) and
            plan['baseline_routes'] == list(OLD_ROUTES) and plan['planned_inputs'] == 32,
            'plan_coverage_disagrees')
    require(plan['height'] == HEIGHT and plan['horizontal_walls'] is None and
            plan['strict_area_tolerance'] == plan['boundary_tolerance'] == EPS and
            plan['reference_area50'] == 50 and plan['sampling'] is False and
            plan['training'] is False and 0 < plan['seconds_per_new_pipeline'] <= 120 and
            type(plan['compaction_sweeps']) is int and 0 <= plan['compaction_sweeps'] <= 16,
            'plan_policy_disagrees')
    require(summary['status'] == 'completed' and summary['planned_inputs'] ==
            summary['completed_inputs'] == len(summary['jobs']) == 32,
            'summary_input_denominator_disagrees')
    jobs = {e['path']: e['sha256'] for e in summary['jobs']}
    require(len(jobs) == 32 and set(jobs) == {'jobs/sample_%03d_%s.json' %
            (e['sample_index'], source) for e in entries for source in SOURCES},
            'job_coverage_disagrees')
    for rel, digest in jobs.items():
        require(meta['output_sha256'].get(rel) == digest, 'job_hash_not_bound', rel)
    rows, baseline_rows, records = [], [], []
    counts = Counter()
    statuses = Counter()
    for entry in entries:
        case = read(frozen/entry['path'])
        for key in ('sample_index', 'candidate_index', 'group'):
            require(case[key] == entry[key], 'case_manifest_identity_disagrees')
        index, n = entry['sample_index'], len(case['polygon_ids'])
        for source, field in SOURCES.items():
            loc = 'sample_%03d_%s' % (index, source)
            job = read(root/('jobs/'+loc+'.json'))
            group = 'failure' if index in FAILURES else 'success'
            base = {'sample_index': index, 'source': source, 'baseline_group': group}
            require(all(job[k] == v for k, v in base.items()) and
                    job['case_sha256'] == entry['sha256'] and
                    job['candidate_index'] == case['candidate_index'] and job['input_field'] == field,
                    'job_identity_disagrees', loc)
            original, cast = case[field], fp32(case[field])
            before = geometry(case, original, loc+'/original')
            cast_geometry = geometry(case, cast, loc+'/cast')
            check_geometry(job['before'], before, loc+'/before')
            old = {}
            oldbrief = {}
            for route in OLD_ROUTES:
                rel = 'jobs/sample_%03d_%s_%s.json' % (index, source, route)
                previous = read(oldroot/rel)
                old[route] = previous
                require(previous['input_actions'] == original and previous['sample_index'] == index
                        and previous['source'] == source, 'legacy_input_disagrees', loc+'/'+route)
                oldbrief[route] = {}
                for precision in ('float64', 'float32'):
                    output = previous['stages'][precision]['actions']
                    actions_checked(output, n, loc+'/'+route+'/'+precision,
                                    stored=precision == 'float32')
                    if precision == 'float32':
                        require(output == fp32(previous['stages']['float64']['actions']),
                                'legacy_not_single_float32_cast', loc+'/'+route)
                    else:
                        require(all(a[2:] == b[2:] for a, b in zip(output, original)),
                                'legacy_rotation_changed', loc+'/'+route)
                    g = geometry(case, output, loc+'/'+route+'/'+precision)
                    check_geometry(previous['stages'][precision]['after'], g,
                                   loc+'/'+route+'/'+precision)
                    baseline_rows.append(base | {'path': 'legacy_'+route, 'status': 'completed',
                        'output_precision': precision, 'after': g, 'comparison': None,
                        'round7_job_sha256': sha(oldroot/rel)})
                    oldbrief[route][precision] = brief(g)
                    counts['baseline_geometries'] += 1
            pipeline = job['pipeline']
            require(pipeline['repair_status'] in ('strict_feasible', 'stalled', 'time_limit',
                    'iteration_limit') and pipeline['compaction_status'] in
                    ('completed', 'skipped_infeasible', 'time_limit', 'not_started'),
                    'pipeline_status_invalid', loc)
            replay = Replay(case, cast, loc+'/pipeline', project_boundary=True)
            require(pipeline['boundary_adjustments'] == replay.boundary_adjustments,
                    'initial_boundary_projection_count_disagrees', loc)
            repair_state, compact_started = None, False
            for move in pipeline['moves']:
                if move['phase'] != 'local_repair' and not compact_started:
                    repair_state = [a[:] for a in replay.actions]
                    compact_started = True
                require(not compact_started or move['phase'] != 'local_repair',
                        'repair_move_after_compaction', loc)
                replay.move(move)
            repair_state = replay.actions if repair_state is None else repair_state
            require(pipeline['repair_actions'] == repair_state and pipeline['actions'] == replay.actions,
                    'pipeline_actions_not_replayed', loc)
            counts.update(replay.counts)
            counts['boundary_adjustments'] += replay.boundary_adjustments
            statuses['new_repair/'+pipeline['repair_status']] += 1
            statuses['new_compact/'+pipeline['compaction_status']] += 1
            saved = fp32(old['repair_only']['repair_actions'])
            saved_geom = geometry(case, saved, loc+'/saved_before')
            check_geometry(job['saved_repair_before'], saved_geom, loc+'/saved_before')
            extra = job['saved_repair_compaction']
            if not saved_geom['strict_strip_valid']:
                require(extra['status'] == 'skipped_infeasible' and extra['moves'] == [] and
                        extra['actions'] == saved, 'saved_infeasible_route_not_preserved', loc)
                counts['saved_routes_skipped_infeasible'] += 1
            else:
                require(extra['status'] in ('completed', 'time_limit'), 'saved_status_invalid', loc)
                saved_replay = Replay(case, saved, loc+'/saved')
                for move in extra['moves']:
                    require(move['phase'] != 'local_repair', 'saved_route_performed_repair', loc)
                    saved_replay.move(move)
                require(extra['actions'] == saved_replay.actions, 'saved_actions_not_replayed', loc)
                counts.update(saved_replay.counts)
                counts['saved_routes_compacted'] += 1
            statuses['saved_compact/'+extra['status']] += 1
            stagebrief, stagegeoms = {}, {}
            for route in ROUTES:
                stage = job['stages'][route]
                output = stage['actions']
                expected_output = {'new_repair': pipeline['repair_actions'],
                    'new_repair_compact': pipeline['actions'],
                    'saved_repair_compact': extra['actions']}[route]
                require(output == expected_output, 'stage_not_pipeline_output', loc+'/'+route)
                actions_checked(output, n, loc+'/'+route, stored=True, rotations=cast)
                require(fp32(output) == output, 'second_storage_cast_changes_result', loc+'/'+route)
                g = geometry(case, output, loc+'/'+route)
                check_geometry(stage['after'], g, loc+'/'+route)
                # Replay comparison arithmetic from separately checked saved
                # metrics, avoiding numerical sign changes at exact contact.
                c = comparison(job['before'], stage['after'], original, output)
                close(stage['comparison'], c, loc+'/'+route+'/comparison', GEOMETRY_ATOL)
                rows.append(base | {'path': route, 'status': 'completed',
                            'output_precision': 'float32', 'after': g, 'comparison': c})
                stagebrief[route], stagegeoms[route] = brief(g), g
                counts['new_stage_geometries'] += 1
            if pipeline['repair_status'] == 'strict_feasible':
                require(stagegeoms['new_repair']['strict_strip_valid'],
                        'repair_success_not_independently_strict', loc)
            if compact_started or pipeline['compaction_status'] == 'completed':
                require(pipeline['repair_status'] == 'strict_feasible' and
                        stagegeoms['new_repair_compact']['strict_strip_valid'],
                        'compaction_success_not_independently_strict', loc)
            changes = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(
                cast, pipeline['actions'])]
            records.append(base | {'piece_count': n, 'before': brief(before),
                'input_after_single_float32_cast': brief(cast_geometry),
                'new_stages': stagebrief, 'legacy_stages': oldbrief,
                'new_repair_status': pipeline['repair_status'],
                'new_compaction_status': pipeline['compaction_status'],
                'saved_compaction_status': extra['status'],
                'boundary_adjustments': replay.boundary_adjustments,
                'new_pipeline_moves': len(pipeline['moves']),
                'saved_compaction_moves': len(extra['moves']),
                'new_actions_coordinate_range': {axis: [min(a[k] for a in pipeline['actions']),
                   max(a[k] for a in pipeline['actions'])] for k, axis in enumerate(('x', 'y'))},
                'new_final_translation_distance_from_cast_input': stats(changes),
                'elapsed_seconds': pipeline['elapsed_seconds']})
            print(json.dumps({'verified': loc, 'new_stages': 3, 'baseline_outputs': 6}), flush=True)
    require(counts['new_stage_geometries'] == 96 and counts['baseline_geometries'] == 192,
            'output_geometry_coverage_incomplete')
    from postprocess_metrics import summarize
    for name, expected_rows in (('rows', rows), ('baseline_rows', baseline_rows)):
        recorded = summary[name]
        key = lambda r: (r['sample_index'], r['source'], r['path'], r['output_precision'])
        lookup = {key(r): r for r in recorded}
        require(len(lookup) == len(recorded) == len(expected_rows), 'row_coverage_disagrees', name)
        for row in expected_rows:
            require(key(row) in lookup, 'missing_expected_row', name)
            actual = lookup[key(row)]
            for k in ('baseline_group', 'status', 'comparison'):
                close(actual[k], row[k], name+'/'+k, GEOMETRY_ATOL)
            check_geometry(actual['after'], row['after'], name+'/after')
            if name == 'baseline_rows':
                require(actual['round7_job_sha256'] == row['round7_job_sha256'],
                        'baseline_row_job_hash_disagrees')
    aggregate, oldaggregate = summarize(rows), summarize(baseline_rows)
    close(summary['aggregates'], aggregate, 'aggregates', GEOMETRY_ATOL)
    close(summary['baseline_aggregates'], oldaggregate, 'baseline_aggregates', GEOMETRY_ATOL)
    paired = []
    for source in SOURCES:
        selected = [r for r in records if r['source'] == source]
        for route in ROUTES:
            for oldroute in OLD_ROUTES:
                deltas = [r['new_stages'][route]['strip_utilization']-
                    r['legacy_stages'][oldroute]['float32']['strip_utilization'] for r in selected]
                paired.append({'source': source, 'new_route': route, 'legacy_route': oldroute,
                    'denominator': len(selected), 'strip_utilization_delta': stats(deltas),
                    'strict_new_valid': sum(r['new_stages'][route]['strict_strip_valid'] for r in selected),
                    'strict_old_valid': sum(r['legacy_stages'][oldroute]['float32']['strict_strip_valid'] for r in selected),
                    'eps50_new_valid': sum(r['new_stages'][route]['eps50_strip_valid'] for r in selected),
                    'eps50_old_valid': sum(r['legacy_stages'][oldroute]['float32']['eps50_strip_valid'] for r in selected),
                    'joint_strict_count': sum(r['new_stages'][route]['strict_strip_valid'] and
                        r['legacy_stages'][oldroute]['float32']['strict_strip_valid'] for r in selected),
                    'caveat': 'Paired utility alone includes infeasible outputs; validity counts are separate.'})
    gt = [r for r in records if r['source'] == 'gt']
    return {'status': 'verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha(Path(__file__).resolve()),
        'scope': 'Frozen 16 development/regression representatives plus GT, 32 inputs; no sampling or training; no independent generalization claim',
        'geometry': 'Independent Shapely affine.rotate/translate and every pair intersection; no solver or interval imports',
        'thresholds': {'height': HEIGHT, 'horizontal_walls': None, 'strict_area': EPS,
                       'boundary': EPS, 'reference_area50': 50.,
                       'numeric_metric_comparison_atol': GEOMETRY_ATOL},
        'hash_inventory': {'frozen_manifest_sha256': MANIFEST_SHA, 'input_files': len(inputs),
            'new_output_files': len(meta['output_sha256']),
            'legacy_output_files': len(oldmeta['output_sha256']),
            'new_inputs_sha256': sha(root/'inputs.json'), 'new_metadata_sha256': sha(root/'metadata.json'),
            'legacy_metadata_sha256': sha(oldroot/'metadata.json'),
            'script_sha256': {n: expected['scripts/'+n] for n in SCRIPT_INPUTS}},
        'coverage': {'planned_inputs': 32, 'verified_jobs': 32, 'new_stage_outputs': 96,
            'legacy_baseline_outputs': 192, 'original_and_cast_inputs_measured': 64,
            'saved_repair_cast_inputs_measured': 32, 'all_denominators_retained': True},
        'ledger_checks': dict(sorted(counts.items())), 'pipeline_status_counts': dict(sorted(statuses.items())),
        'new_aggregates': aggregate_brief(aggregate), 'legacy_aggregates': aggregate_brief(oldaggregate),
        'gt_regression': {route: {'denominator': 16,
            'strict_valid': sum(r['new_stages'][route]['strict_strip_valid'] for r in gt),
            'strict_regressions_from_original_gt': sum(r['before']['strict_strip_valid'] and
                not r['new_stages'][route]['strict_strip_valid'] for r in gt),
            'boundary_failures': sum(r['new_stages'][route]['height_violation'] > EPS for r in gt)} for route in ROUTES},
        'float32_boundary_checks': {route: {'denominator': 32,
            'boundary_failures': sum(r['new_stages'][route]['height_violation'] > EPS for r in records)} for route in ROUTES},
        'paired_utilization_deltas_vs_saved_float32': paired, 'samples': records,
        'limits': ['Accepted endpoint feasibility is checked; continuous collision-free travel is not required.',
                   'The verifier does not rerun the optimizer or claim global optimality.',
                   'Legacy official Linux metric vectors remain archived; this run independently verifies reference contours.',
                   'Positive-area pair counts below 1e-9 can depend on affine roundoff; threshold counts and unthresholded totals are checked.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'round7', 'results', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'verification_output_already_exists')
    try:
        report = verify(args)
    except VerificationError as exc:
        print(json.dumps({'status': 'verification_failed', 'code': exc.code,
                          'location': exc.location}), flush=True)
        raise SystemExit(1)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], **report['coverage'],
                      'ledger_moves': report['ledger_checks'].get('moves', 0)}), flush=True)


if __name__ == '__main__':
    main()
