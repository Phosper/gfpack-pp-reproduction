#!/usr/bin/env python3
"""Verify two-start compaction provenance, stored endpoints, and seed ledgers.

Uses the independent affine geometry and struct storage helpers from the
one-start verifier, never the solver, triangulation, or interval implementation.
No complete actions, contours, or machine paths are written to the public JSON.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import verify_compact_layouts as independent

require = independent.require
read = independent.read
sha = independent.sha
close = independent.close
geometry = independent.geometry
check_geometry = independent.check_geometry
fp32 = independent.fp32
EPS = independent.EPS
HEIGHT = independent.HEIGHT
ATOL = independent.GEOMETRY_ATOL
SCRIPTS = ('compact_portfolio.py', 'run_compact_portfolio.py', 'compact_layout.py',
           'translation_intervals.py', 'postprocess_metrics.py',
           'run_compact_layouts.py', 'check_collision_saved_cases.py')


def inventory(directory):
    metadata = read(directory/'metadata.json')
    require(metadata['status'] == 'completed' and metadata['changed_inputs'] == [],
            'incomplete_or_changed_input_evidence')
    files = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
             if p.is_file() and p != directory/'metadata.json'}
    require(files == set(metadata['output_sha256']), 'output_inventory_incomplete')
    for rel, digest in metadata['output_sha256'].items():
        require(sha(independent.contained(directory, rel)) == digest,
                'output_hash_disagrees', rel)
    return metadata


def replay_pipeline(case, start, result, location):
    """Full seed replay includes explicit initial representable-y projection."""
    count = len(case['polygon_ids'])
    cast = fp32(start)
    replay = independent.Replay(case, cast, location, project_boundary=True)
    require(result['boundary_adjustments'] == replay.boundary_adjustments,
            'seed_boundary_projection_count_disagrees', location)
    require(result['repair_status'] in ('strict_feasible', 'stalled', 'time_limit',
            'iteration_limit') and result['compaction_status'] in
            ('completed', 'skipped_infeasible', 'time_limit', 'not_started'),
            'seed_status_invalid', location)
    repair_state, compact_started = None, False
    for move in result['moves']:
        if move['phase'] != 'local_repair' and not compact_started:
            repair_state = [a[:] for a in replay.actions]
            compact_started = True
        require(not compact_started or move['phase'] != 'local_repair',
                'seed_repair_move_after_compaction', location)
        replay.move(move)
    repair_state = replay.actions if repair_state is None else repair_state
    require(result['repair_actions'] == repair_state and result['actions'] == replay.actions,
            'seed_result_not_ledger_endpoint', location)
    for field in ('repair_actions', 'actions'):
        independent.actions_checked(result[field], count, location+'/'+field,
                                    stored=True, rotations=cast)
    repaired = geometry(case, result['repair_actions'], location+'/repair_endpoint')
    final = geometry(case, result['actions'], location+'/final_endpoint')
    if result['repair_status'] == 'strict_feasible':
        require(repaired['strict_strip_valid'], 'seed_repair_success_not_strict', location)
    if compact_started or result['compaction_status'] == 'completed':
        require(result['repair_status'] == 'strict_feasible' and
                final['strict_strip_valid'], 'seed_compaction_success_not_strict', location)
    require(math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] >= 0,
            'seed_elapsed_time_invalid', location)
    counters = replay.counts.copy()
    counters['boundary_adjustments'] += replay.boundary_adjustments
    counters['pipelines_replayed'] += 1
    return final, repaired, counters


def paired_summary(records, source, reference):
    selected = [r for r in records if r['source'] == source]
    first = [r['selected'] for r in selected]
    other = [r['references'][reference] for r in selected]
    strict = [i for i, (a, b) in enumerate(zip(first, other))
              if a['strict_strip_valid'] and b['strict_strip_valid']]
    area50 = [i for i, (a, b) in enumerate(zip(first, other))
              if a['eps50_strip_valid'] and b['eps50_strip_valid']]
    delta = [a['strip_utilization']-b['strip_utilization'] for a, b in zip(first, other)]
    worst = min(range(len(delta)), key=lambda i: (delta[i], selected[i]['sample_index']))
    result = {'source': source, 'reference': reference, 'denominator': len(selected),
        'all_paired_strip_utilization_delta': independent.stats(delta),
        'joint_strict_valid_count': len(strict),
        'joint_strict_strip_utilization_delta': independent.stats([delta[i] for i in strict]),
        'joint_area50_valid_count': len(area50),
        'joint_area50_strip_utilization_delta': independent.stats([delta[i] for i in area50]),
        'selected_strict_valid': sum(g['strict_strip_valid'] for g in first),
        'reference_strict_valid': sum(g['strict_strip_valid'] for g in other),
        'selected_area50_valid': sum(g['eps50_strip_valid'] for g in first),
        'reference_area50_valid': sum(g['eps50_strip_valid'] for g in other),
        'worst_all_paired_delta': {'sample_index': selected[worst]['sample_index'],
                                  'delta': delta[worst]},
        'mean_strip_utilization_all': {
            'selected': math.fsum(g['strip_utilization'] for g in first)/len(selected),
            'reference': math.fsum(g['strip_utilization'] for g in other)/len(selected)}}
    for label, criterion in (('strict', 'strict_strip_valid'), ('area50', 'eps50_strip_valid')):
        result['mean_'+label+'_valid_utilization_failures_zero'] = {
            'selected': math.fsum(g['strip_utilization'] for g in first if g[criterion])/len(selected),
            'reference': math.fsum(g['strip_utilization'] for g in other if g[criterion])/len(selected)}
    return result


def verify(args):
    frozen = args.frozen.resolve()
    oldroot, rawroot, root = args.round7.resolve(), args.raw_results.resolve(), args.portfolio.resolve()
    scriptroot = Path(__file__).resolve().parent
    manifest = read(frozen/'manifest.json')
    require(sha(frozen/'manifest.json') == independent.MANIFEST_SHA and
            (frozen/'manifest.sha256').read_text().split()[0] == independent.MANIFEST_SHA,
            'frozen_manifest_changed')
    entries = manifest['cases']
    require(len(entries) == 16 and sorted(e['sample_index'] for e in entries) ==
            list(independent.SAMPLES), 'frozen_case_coverage')
    oldmeta, rawmeta, metadata = inventory(oldroot), inventory(rawroot), inventory(root)
    rawinputs = read(rawroot/'inputs.json')
    expected_raw = {'scripts/'+name: sha(scriptroot/name) for name in independent.SCRIPT_INPUTS}
    expected_raw.update({'frozen/manifest.json': independent.MANIFEST_SHA,
                         'round7/metadata.json': sha(oldroot/'metadata.json')})
    for entry in entries:
        index = entry['sample_index']
        require(entry['path'] == 'cases/sample_%03d.json' % index and entry['group'] ==
                ('failure_development' if index in independent.FAILURES else 'success_regression'),
                'frozen_case_identity')
        require(sha(independent.contained(frozen, entry['path'])) == entry['sha256'],
                'frozen_case_hash_disagrees')
        expected_raw['frozen/'+entry['path']] = entry['sha256']
    expected_raw.update({'round7/'+rel: value for rel, value in oldmeta['output_sha256'].items()})
    require(rawinputs == expected_raw and rawmeta['input_count'] == len(expected_raw),
            'raw_input_inventory_disagrees')
    expected = {'scripts/'+name: sha(scriptroot/name) for name in SCRIPTS}
    expected.update({'frozen/manifest.json': independent.MANIFEST_SHA,
                     'raw/metadata.json': sha(rawroot/'metadata.json')})
    expected.update({'raw/'+rel: value for rel, value in rawmeta['output_sha256'].items()})
    expected.update({'frozen/'+e['path']: e['sha256'] for e in entries})
    for entry in entries:
        for source in independent.SOURCES:
            rel = 'jobs/sample_%03d_%s_repair_only.json' % (entry['sample_index'], source)
            expected['round7/'+rel] = oldmeta['output_sha256'][rel]
    require(read(root/'inputs.json') == expected and metadata['input_count'] == len(expected),
            'portfolio_input_inventory_disagrees')
    plan, rawplan = read(root/'plan.json'), read(rawroot/'plan.json')
    require(plan['planned_inputs'] == rawplan['planned_inputs'] == 32 and
            plan['height'] == rawplan['height'] == HEIGHT and
            plan['horizontal_walls'] is rawplan['horizontal_walls'] is None and
            plan['original_one_start_results_retained'] is True,
            'portfolio_plan_geometry_or_scope_disagrees')
    require(plan['seconds_per_attempt'] == rawplan['seconds_per_new_pipeline'] == 45. and
            plan['sweeps_per_attempt'] == rawplan['compaction_sweeps'] == 4 and
            rawplan['sampling'] is False and rawplan['training'] is False and
            rawplan['sources'] == independent.SOURCES and
            rawplan['strict_area_tolerance'] == rawplan['boundary_tolerance'] == EPS and
            rawplan['reference_area50'] == 50., 'portfolio_plan_policy_disagrees')
    summary, rawsummary = read(root/'summary.json'), read(rawroot/'summary.json')
    jobpaths = {'jobs/sample_%03d_%s.json' % (e['sample_index'], source)
                for e in entries for source in independent.SOURCES}
    for summ, meta in ((summary, metadata), (rawsummary, rawmeta)):
        require(summ['status'] == 'completed' and summ['planned_inputs'] ==
                summ['completed_inputs'] == len(summ['jobs']) == 32,
                'input_denominator_incomplete')
        jobs = {j['path']: j['sha256'] for j in summ['jobs']}
        require(len(jobs) == 32 and set(jobs) == jobpaths and all(
                meta['output_sha256'][rel] == value for rel, value in jobs.items()),
                'job_hash_coverage_disagrees')
    rows, records, counts, states = [], [], Counter(), Counter()
    raw_rows = {(r['sample_index'], r['source'], r['path']): r for r in rawsummary['rows']}
    require(len(raw_rows) == len(rawsummary['rows']) == 96, 'raw_stage_row_coverage_incomplete')
    for entry in entries:
        case = read(frozen/entry['path'])
        for key in ('sample_index', 'candidate_index', 'group'):
            require(case[key] == entry[key], 'frozen_case_manifest_identity_disagrees')
        index, n = entry['sample_index'], len(case['polygon_ids'])
        for source, field in independent.SOURCES.items():
            stem = 'sample_%03d_%s' % (index, source)
            rel = 'jobs/'+stem+'.json'
            oldrel = 'jobs/'+stem+'_repair_only.json'
            job, rawjob, seedjob = read(root/rel), read(rawroot/rel), read(oldroot/oldrel)
            base = {'sample_index': index, 'source': source,
                    'baseline_group': 'failure' if index in independent.FAILURES else 'success'}
            require(all(job[k] == rawjob[k] == v for k, v in base.items()) and
                    job['case_sha256'] == rawjob['case_sha256'] == entry['sha256'] and
                    job['candidate_index'] == rawjob['candidate_index'] == case['candidate_index'],
                    'portfolio_raw_identity_disagrees', stem)
            require(rawjob['input_field'] == field and seedjob['input_field'] == field and
                    seedjob['sample_index'] == index and seedjob['source'] == source and
                    seedjob['case_sha256'] == entry['sha256'] and
                    seedjob['candidate_index'] == case['candidate_index'] and
                    seedjob['input_actions'] == case[field], 'seed_not_same_frozen_source', stem)
            require(job['raw_job_sha256'] == rawmeta['output_sha256'][rel] and
                    job['legacy_seed_job_sha256'] == oldmeta['output_sha256'][oldrel],
                    'portfolio_provenance_hash_disagrees', stem)
            original, rawcast = case[field], fp32(case[field])
            before = geometry(case, original, stem+'/original')
            check_geometry(rawjob['before'], before, stem+'/raw_before')
            seed = fp32(seedjob['repair_actions'])
            independent.actions_checked(seed, n, stem+'/seed', stored=True, rotations=rawcast)
            require(all(a[2:] == b[2:] for a, b in zip(seedjob['repair_actions'], original)),
                    'legacy_seed_changed_rotation', stem)
            seed_before = geometry(case, seed, stem+'/seed_before')
            raw_result = rawjob['pipeline']
            raw_actions = raw_result['actions']
            independent.actions_checked(raw_actions, n, stem+'/raw_final', stored=True, rotations=rawcast)
            raw_geom = geometry(case, raw_actions, stem+'/raw_final')
            require(rawjob['stages']['new_repair_compact']['actions'] == raw_actions,
                    'raw_final_not_raw_stage', stem)
            check_geometry(rawjob['stages']['new_repair_compact']['after'], raw_geom,
                           stem+'/raw_stage')
            raw_row = raw_rows[(index, source, 'new_repair_compact')]
            require(raw_row['baseline_group'] == base['baseline_group'] and
                    raw_row['output_precision'] == 'float32' and raw_row['status'] == 'completed',
                    'raw_summary_final_identity_disagrees', stem)
            check_geometry(raw_row['after'], raw_geom, stem+'/raw_summary')
            result = job['portfolio']
            duplicated = rawcast == seed
            require(type(result['seed_reused_raw_result']) is bool and
                    result['seed_reused_raw_result'] == duplicated, 'seed_duplicate_reuse_disagrees', stem)
            if duplicated:
                require(result['seed_result'] is None, 'duplicate_seed_was_recomputed', stem)
                seed_result, seed_geom = raw_result, raw_geom
                counts['duplicate_starts_reused'] += 1
            else:
                require(isinstance(result['seed_result'], dict), 'distinct_seed_result_missing', stem)
                seed_result = result['seed_result']
                seed_geom, repaired, checked = replay_pipeline(case, seed, seed_result, stem+'/seed')
                counts.update(checked)
                counts['seed_repair_endpoint_geometries'] += 1
                counts['distinct_seed_attempts'] += 1
            attempts = result['attempts']
            require(isinstance(attempts, list) and len(attempts) == 2 and
                    [r['start'] for r in attempts] == ['raw', 'legacy_repair'],
                    'attempt_coverage_or_order_disagrees', stem)
            for attempt, attempt_result, measured in zip(attempts,
                    (raw_result, seed_result), (raw_geom, seed_geom)):
                check_geometry(attempt['after'], measured, stem+'/attempt_'+attempt['start'])
                require(attempt['repair_status'] == attempt_result['repair_status'] and
                        attempt['compaction_status'] == attempt_result['compaction_status'],
                        'attempt_status_not_solver_status', stem)
            feasible = [i for i, g in enumerate((raw_geom, seed_geom)) if g['strict_strip_valid']]
            # Exact stored metric ties must favor raw. Independently measured
            # spans are also checked against the selected minimum below.
            chosen = min(feasible, key=lambda i: (attempts[i]['after']['span_x'], i)) if feasible else None
            expected_start = ('raw', 'legacy_repair')[chosen] if chosen is not None else None
            require(result['selected_start'] == expected_start and result['status'] ==
                    ('strict_feasible' if chosen is not None else 'no_strict_feasible_attempt'),
                    'selection_not_strict_minimum_span', stem)
            selected_result = raw_result if chosen in (None, 0) else seed_result
            require(result['actions'] == selected_result['actions'], 'selected_actions_not_chosen_attempt', stem)
            output = result['actions']
            independent.actions_checked(output, n, stem+'/selected', stored=True, rotations=rawcast)
            require(fp32(output) == output, 'selected_second_float32_cast_changes_actions', stem)
            selected_geom = geometry(case, output, stem+'/selected')
            check_geometry(job['after'], selected_geom, stem+'/job_after')
            if chosen is not None:
                require(selected_geom['strict_strip_valid'] and selected_geom['span_x'] <=
                        min((raw_geom, seed_geom)[i]['span_x'] for i in feasible)+ATOL,
                        'selected_not_independent_feasible_minimum', stem)
            if raw_geom['strict_strip_valid']:
                require(selected_geom['strict_strip_valid'] and
                        selected_geom['span_x'] <= raw_geom['span_x']+EPS,
                        'portfolio_regressed_feasible_raw_span', stem)
            # If raw is infeasible, feasibility has priority over its narrower
            # illegal span; retain and report that possible expansion.
            timing = raw_result['elapsed_seconds']+(0. if duplicated else seed_result['elapsed_seconds'])
            close(result['new_solver_seconds_without_legacy_seed_generation'], timing,
                  stem+'/timing', 1e-9)
            c = independent.comparison(rawjob['before'], job['after'], original, output)
            close(job['comparison'], c, stem+'/comparison', ATOL)
            require(job['path'] == 'two_start_compact' and job['status'] == 'completed' and
                    job['output_precision'] == 'float32', 'selected_row_identity_disagrees', stem)
            rows.append(base | {'path': 'two_start_compact', 'status': 'completed',
                        'output_precision': 'float32', 'after': selected_geom, 'comparison': c})
            references = {'raw_one_start': independent.brief(raw_geom),
                          'original_input': independent.brief(before)}
            for route in independent.OLD_ROUTES:
                legacy = read(oldroot/('jobs/'+stem+'_'+route+'.json'))
                require(legacy['sample_index'] == index and legacy['source'] == source and
                        legacy['input_actions'] == original and legacy['case_sha256'] == entry['sha256'],
                        'legacy_comparison_not_same_input', stem+'/'+route)
                old_actions = legacy['stages']['float32']['actions']
                independent.actions_checked(old_actions, n, stem+'/'+route, stored=True,
                                            rotations=rawcast)
                g = geometry(case, old_actions, stem+'/'+route)
                check_geometry(legacy['stages']['float32']['after'], g, stem+'/'+route)
                references['legacy_'+route] = independent.brief(g)
                counts['legacy_comparison_geometries'] += 1
            translation = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(rawcast, output)]
            records.append(base | {'piece_count': n, 'selected_start': expected_start,
                'status': result['status'], 'seed_reused_raw_result': duplicated,
                'seed_before': independent.brief(seed_before),
                'seed_after': independent.brief(seed_geom), 'selected': independent.brief(selected_geom),
                'references': references,
                'selected_translation_from_cast_raw': independent.stats(translation),
                'selected_coordinate_range': {axis: [min(a[k] for a in output),
                    max(a[k] for a in output)] for k, axis in enumerate(('x', 'y'))},
                'new_solver_seconds_without_legacy_seed_generation': timing,
                'distinct_seed_repair_status': None if duplicated else seed_result['repair_status'],
                'distinct_seed_compaction_status': None if duplicated else seed_result['compaction_status']})
            states[source+'/'+str(expected_start)] += 1
            counts['selected_geometries'] += 1
            counts['raw_final_geometries'] += 1
            counts['seed_input_geometries'] += 1
            counts['seed_final_geometries'] += 1
            print(json.dumps({'verified': stem, 'selected': expected_start,
                              'seed_reused': duplicated}), flush=True)
    from postprocess_metrics import summarize
    require(len(rows) == 32 and Counter(r['source'] for r in rows) == {'pred': 16, 'gt': 16},
            'selected_denominators_incomplete')
    lookup = {(r['sample_index'], r['source']): r for r in summary['rows']}
    require(len(lookup) == len(summary['rows']) == 32, 'selected_summary_rows_incomplete')
    for row in rows:
        actual = lookup[(row['sample_index'], row['source'])]
        for key in ('baseline_group', 'path', 'status', 'output_precision', 'comparison'):
            close(actual[key], row[key], 'summary_rows/'+key, ATOL)
        check_geometry(actual['after'], row['after'], 'summary_rows/after')
    aggregate = summarize(rows)
    close(summary['aggregates'], aggregate, 'selected_aggregates', ATOL)
    gt = [r for r in records if r['source'] == 'gt']
    references = ('raw_one_start', 'original_input', 'legacy_gap_only',
                  'legacy_repair_only', 'legacy_repair_then_gap')
    feasible_raw = [r for r in records if r['references']['raw_one_start']['strict_strip_valid']]
    require(all(r['selected']['span_x'] <= r['references']['raw_one_start']['span_x']+EPS
                for r in feasible_raw), 'paired_raw_span_regression')
    return {'status': 'verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': sha(Path(__file__).resolve()),
        'independent_geometry_helper_sha256': sha(Path(independent.__file__).resolve()),
        'scope': 'Uniform two-start postprocessing of the same 16 frozen predictions and GT; 32 inputs, no new candidate selection or training',
        'independence': 'All final raw, seed, selected and paired legacy endpoints reconstructed with affine geometry; every distinct seed accepted move independently replayed against all other contours',
        'hashes': {'frozen_manifest_sha256': independent.MANIFEST_SHA,
            'raw_metadata_sha256': sha(rawroot/'metadata.json'),
            'legacy_metadata_sha256': sha(oldroot/'metadata.json'),
            'portfolio_inputs_sha256': sha(root/'inputs.json'),
            'portfolio_metadata_sha256': sha(root/'metadata.json'),
            'script_sha256': {name: expected['scripts/'+name] for name in SCRIPTS}},
        'coverage': {'planned_inputs': 32, 'predictions': 16, 'gt': 16,
            'selected_outputs_checked': 32, 'attempts_checked': 64,
            'distinct_seed_attempts': counts['distinct_seed_attempts'],
            'duplicate_starts_reused': counts['duplicate_starts_reused'],
            'raw_output_hashes_checked': len(rawmeta['output_sha256']),
            'legacy_output_hashes_checked': len(oldmeta['output_sha256']),
            'portfolio_output_hashes_checked': len(metadata['output_sha256']),
            'portfolio_inputs_checked': len(expected), 'all_denominators_retained': True},
        'thresholds': {'height': HEIGHT, 'horizontal_walls': None, 'strict_area': EPS,
            'boundary': EPS, 'reference_area50': 50., 'metric_comparison_atol': ATOL},
        'selection': 'Only strict stored endpoints eligible, minimum span, raw wins exact stored metric ties; all-failed inputs retain explicit failure and raw partial endpoint',
        'selection_counts': dict(sorted(states.items())), 'ledger_checks': dict(sorted(counts.items())),
        'paired_raw_span_nonregression': {'eligible_strict_raw_inputs': len(feasible_raw),
            'regressions': sum(r['selected']['span_x'] > r['references']['raw_one_start']['span_x']+EPS
                               for r in feasible_raw)},
        'aggregate': independent.aggregate_brief(aggregate),
        'gt_regression': {'denominator': 16,
            'strict_valid': sum(r['selected']['strict_strip_valid'] for r in gt),
            'strict_regressions_from_original': sum(r['references']['original_input']['strict_strip_valid']
                and not r['selected']['strict_strip_valid'] for r in gt),
            'boundary_failures': sum(r['selected']['height_violation'] > EPS for r in gt)},
        'final_float32_boundary_failures': sum(r['selected']['height_violation'] > EPS for r in records),
        'paired_statistics': [paired_summary(records, source, reference)
                              for source in independent.SOURCES for reference in references],
        'samples': records,
        'limitations': ['Frozen development cases, not independent generalization or a 512-candidate rate.',
            'Already archived raw attempt ledgers have a separate one-start verifier; this verifier remeasures their endpoints and replays every additional distinct seed ledger.',
            'Endpoint feasibility does not imply collision-free continuous motion.',
            'Feasibility outranks the narrower span of an infeasible raw attempt.',
            'Solver timing excludes previously archived legacy seed generation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'round7', 'raw-results', 'portfolio', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'verification_output_already_exists')
    try:
        report = verify(args)
    except independent.VerificationError as exc:
        print(json.dumps({'status': 'verification_failed', 'code': exc.code,
                          'location': exc.location}), flush=True)
        raise SystemExit(1)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], **report['coverage']}), flush=True)


if __name__ == '__main__':
    main()
