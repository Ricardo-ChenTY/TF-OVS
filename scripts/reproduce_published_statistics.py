"""Recompute published E4 intervals and CLIPtrase metrics using repository evidence only."""
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
from summarize_proposal_e4_expanded import bootstrap, interval, METHODS, TARGETS, write

ROOT = Path(__file__).resolve().parents[1]

def checked_archive(path):
    expected = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Archive checksum mismatch: ' + path.name)
    return tarfile.open(path)

def compare(actual, expected, keys, numbers):
    index = {tuple(row[k] for k in keys): row for row in expected}
    if len(index) != len(expected) or len(actual) != len(expected):
        raise ValueError('Duplicate or missing table rows')
    for row in actual:
        old = index.pop(tuple(row[k] for k in keys))
        for name in numbers:
            if not np.isclose(float(row[name]), float(old[name]), rtol=0, atol=1e-12):
                raise ValueError(f'{keys}: {row}: mismatch in {name}')
    if index:
        raise ValueError('Unmatched table rows')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / 'docs/rebuttal/evidence'
    table, paired, arrays, ordered_ids = [], [], {}, {}
    with checked_archive(ROOT / 'docs/rebuttal/artifacts/e4_expanded_results_20261016.tar.gz') as tar:
        members = {m.name: m for m in tar if m.isfile()}
        for method in METHODS + ['corrclip_online']:
            values = []
            for ds in TARGETS:
                suffix = f'/jobs/{method}/{ds}/per_image.jsonl'
                matches = [n for n in members if n.endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError('Missing/ambiguous timings: ' + suffix)
                path = matches[0]
                rows = [json.loads(s) for s in tar.extractfile(members[path])]
                ids = [r['image_id'] for r in rows]
                if len(rows) != 1000 or len(set(ids)) != 1000:
                    raise ValueError('Expected exactly 1000 unique images')
                if ds in ordered_ids and ids != ordered_ids[ds]:
                    raise ValueError('Methods used different images or order')
                ordered_ids[ds] = ids
                a = np.array([r['online_seconds'] for r in rows])
                if not np.isfinite(a).all() or (a <= 0).any():
                    raise ValueError('Invalid timings')
                check = json.load(tar.extractfile(members[path.replace('per_image.jsonl', 'prediction_pairing.json')]))
                if check['images'] != 1000 or check['exact_matches'] != 1000:
                    raise ValueError('Incomplete published prediction comparison')
                arrays[method, ds] = a
                values.append(a)
            per, macro = bootstrap(values)
            label = 'corrclip_online_sam2_8' if method == 'corrclip_online' else method
            for ds, a, b in list(zip(TARGETS, values, per)) + [('equal_target_mean', np.concatenate(values), macro)]:
                lo, hi = interval(b)
                table.append(dict(method=label, target=ds, images=len(a), online_mean_seconds=float(a.mean()), ci95_low_seconds=lo, ci95_high_seconds=hi))
        for suffix in ['clip', 'siglip']:
            values = [arrays['dinov2_sam_' + suffix, ds] - arrays['sam_amg_' + suffix, ds] for ds in TARGETS]
            per, macro = bootstrap(values)
            for ds, a, b in list(zip(TARGETS, values, per)) + [('equal_target_mean', np.concatenate(values), macro)]:
                lo, hi = interval(b)
                paired.append(dict(naming=suffix, target=ds, paired_images=len(a), dino_minus_sam_mean_seconds=float(a.mean()), ci95_low_seconds=lo, ci95_high_seconds=hi))
    def csv_rows(path):
        with path.open() as f:
            return list(csv.DictReader(f))
    e4 = evidence / 'e4_expanded_results_20261016'
    compare(table, csv_rows(e4 / 'latency_ci_five_configurations.csv'), ['method', 'target'], ['images', 'online_mean_seconds', 'ci95_low_seconds', 'ci95_high_seconds'])
    compare(paired, csv_rows(e4 / 'paired_dino_overhead_ci.csv'), ['naming', 'target'], ['paired_images', 'dino_minus_sam_mean_seconds', 'ci95_low_seconds', 'ci95_high_seconds'])
    write(args.out / 'latency_ci.csv', table)
    write(args.out / 'paired_dino_overhead_ci.csv', paired)
    clip = evidence / 'cliptrase_native_crop_20261015'
    scores = []
    with np.load(clip / 'confusion_matrices.npz', allow_pickle=False) as data:
        for ds in ['ade847', 'context59', 'context459', 'ade150']:
            cm = data[ds]
            tp = np.diag(cm); gt = cm.sum(1); pred = cm.sum(0); union = gt + pred - tp
            iou = np.divide(tp, union, out=np.zeros(len(tp)), where=union > 0)
            scores.append(dict(dataset=ds,
                native_gt_supported_miou_percent=float((tp / (union + .01))[gt > 0].mean() * 100),
                gt_supported_miou_no_epsilon_percent=float(iou[gt > 0].mean() * 100),
                union_supported_miou_percent=float(iou[union > 0].mean() * 100),
                ziou_gt_percent=float((tp[gt > 0] == 0).mean() * 100),
                ziou_union_percent=float((tp[union > 0] == 0).mean() * 100),
                gt_supported_classes=int((gt > 0).sum()), union_supported_classes=int((union > 0).sum())))
    compare(scores, csv_rows(clip / 'scores.csv'), ['dataset'], list(scores[0])[1:])
    write(args.out / 'cliptrase_scores.csv', scores)
    report = dict(status='passed', timing_records=20000, latency_rows=25, paired_rows=10, cliptrase_score_rows=4,
                  tolerance=1e-12, scope='CPU reaggregation of published measurements and confusion matrices; not fresh inference or independent validation of their collection.')
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
