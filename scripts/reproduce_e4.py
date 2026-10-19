"""Relocate the frozen five-configuration E4 worker without editing its model code.

prepare verifies published source/weights and all selected input bytes; run executes
one method/target and compares output-array hashes with the published measurements.
No historical run directories or historical prediction PNGs are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def relocate(path, mapping):
    path = Path(path)
    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        return Path(new).expanduser().resolve() / relative
    raise ValueError('No path mapping for ' + str(path))

def archive(name):
    p = ROOT / 'docs/rebuttal/artifacts' / name
    if sha(p) != p.with_suffix(p.suffix + '.sha256').read_text().split()[0]:
        raise ValueError('Archive checksum mismatch: ' + name)
    return tarfile.open(p)

def prepare(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    run = args.run.resolve()
    if run.exists():
        raise ValueError('Use a new output directory; refusing to overwrite')
    group = args.group
    name = group + '_e4_expanded_protocol_20261015.tar.gz'
    with archive(name) as tar:
        files = {Path(m.name).relative_to(Path(m.name).parts[0]).as_posix(): tar.extractfile(m).read()
                 for m in tar if m.isfile()}
    protocol = json.loads(files['protocol.json'])
    sources = json.loads(files['source_sha256.json'])
    run.mkdir(parents=True)
    # Copy only ordinary code files, never tar links or arbitrary archive paths.
    for name, blob in files.items():
        if name.startswith('code/'):
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive member')
            dest = run / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
    mapping = config['path_map']
    frozen = {}
    for old, digest in sources.items():
        if '/code/' in old and '/runs/' in old:
            path = run / 'code' / old.split('/code/', 1)[1]
        else:
            path = relocate(old, mapping)
        if sha(path) != digest:
            raise ValueError('Source/asset differs from measured run: ' + str(path))
        frozen[str(path)] = digest
    checked = 0
    for ds, spec in protocol['datasets'].items():
        for key, digest_key in [('manifest', 'sha256'), ('warmup_manifest', 'warmup_sha256')]:
            name = spec[key]
            if hashlib.sha256(files[name]).hexdigest() != spec[digest_key]:
                raise ValueError('Original manifest checksum mismatch')
            rows = [json.loads(line) for line in files[name].splitlines()]
            for row in rows:
                for field, digest_field in [('image_path', 'image_sha256'), ('mask_path', 'mask_sha256')]:
                    path = relocate(row[field], mapping)
                    if sha(path) != row[digest_field]:
                        raise ValueError('Input content mismatch: ' + str(path))
                    row[field] = str(path)
                checked += 1
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            spec[digest_key] = sha(target)
    # Old reference PNG paths and queue paths are provenance only, not dependencies.
    save(run / 'original_protocol.json', json.loads(files['protocol.json']))
    for key in ['references_sha256', 'primary_run', 'primary_protocol_sha256']:
        protocol.pop(key, None)
    protocol['reproduction'] = dict(parent_archive=group + '_e4_expanded_protocol_20261015.tar.gz',
                                   input_images_checked=checked, comparison='published raw_prediction_sha256; array dtype and pixels must both match')
    frozen[str(Path(__file__).resolve())] = sha(__file__)
    save(run / 'runtime.json', config)
    frozen[str(run / 'runtime.json')] = sha(run / 'runtime.json')
    save(run / 'source_sha256.json', frozen)
    save(run / 'protocol.json', protocol)
    expected = {}
    with archive('e4_expanded_results_20261016.tar.gz') as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith('/per_image.jsonl') or '/jobs/' not in m.name:
                continue
            method, ds, _ = m.name.split('/jobs/', 1)[1].split('/')
            if method in protocol['methods']:
                expected.setdefault(method, {})[ds] = {r['image_id']: r['raw_prediction_sha256']
                    for r in (json.loads(line) for line in tar.extractfile(m))}
    save(run / 'expected_predictions.json', expected)
    frozen[str(run / 'expected_predictions.json')] = sha(run / 'expected_predictions.json')
    save(run / 'source_sha256.json', frozen)
    print(json.dumps(dict(status='prepared', sources=len(frozen), input_images=checked, run=str(run))))

def execute(args):
    run = args.run.resolve()
    config = json.loads((run / 'runtime.json').read_text())
    # Fail before importing models if user shell settings would silently change recipes.
    for key in list(os.environ):
        if key.startswith('TF_OVOS_'):
            del os.environ[key]
    allowed = {'TF_OVOS_SAM_CHECKPOINT', 'TF_OVOS_CLIP_CACHE', 'TF_OVOS_HF_CACHE', 'TF_OVOS_DINOV2_CACHE',
               'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'MPLCONFIGDIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'}
    if set(config['environment']) - allowed:
        raise ValueError('Only asset/cache environment overrides are permitted')
    os.environ.update(config['environment'])
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '4'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(run / 'code'), str(run / 'code/src')] + config.get('extra_python_paths', [])
    import e4_full_common
    e4_full_common.ROOT = Path(config['repository']).resolve()
    import measure_e4
    measure_e4.ROOT = e4_full_common.ROOT
    measure_e4.LEGACY = Path(config['official_methods']).resolve()
    measure_e4.CACHE = Path(config['e4_cache']).resolve()
    # Check that the vocabulary actually loaded by ROOT matches the frozen snapshot.
    original = json.loads((run / 'original_protocol.json').read_text())
    if args.method not in original['methods']:
        raise ValueError('Method is outside this frozen protocol')
    source = json.loads((run / 'source_sha256.json').read_text())
    source_resolved = {str(Path(path).resolve()): digest for path, digest in source.items()}
    env = config['environment']
    if args.method == 'corrclip_online':
        for path in [measure_e4.LEGACY / 'CorrCLIP/corrclip_segmentor.py',
                     measure_e4.CACHE / 'weights/sam2_hiera_large.pt',
                     measure_e4.CACHE / 'clip/b16_fullcc2.5b.pt',
                     Path(env['TORCH_HOME']) / 'hub/checkpoints/dino_vitbase8_pretrain.pth']:
            if str(path.resolve()) not in source_resolved:
                raise ValueError('Unverified CorrCLIP asset: ' + str(path))
    else:
        vocab = measure_e4.ROOT / 'configs/vocab' / measure_e4.DS[args.dataset][1]
        if str(vocab) not in source or sha(vocab) != source[str(vocab)]:
            raise ValueError('Runtime vocabulary is not the verified vocabulary')
        checkpoint = str(Path(env['TF_OVOS_SAM_CHECKPOINT']).resolve())
        if checkpoint not in source_resolved:
            raise ValueError('SAM checkpoint was not verified')
        for key in ['TF_OVOS_CLIP_CACHE', 'TF_OVOS_HF_CACHE', 'TF_OVOS_DINOV2_CACHE']:
            cache = Path(env[key]).resolve()
            if not any(Path(path).is_relative_to(cache) for path in source_resolved):
                raise ValueError('Unverified model cache: ' + key)
    import tf_ovos
    if not Path(tf_ovos.__file__).resolve().is_relative_to(run / 'code/src'):
        raise ValueError('Imported an unfrozen package')
    import measure_e4_full
    sys.argv = ['measure_e4_full', '--run', str(run), '--method', args.method, '--dataset', args.dataset]
    if args.stop_after:
        sys.argv += ['--stop-after', str(args.stop_after)]
    measure_e4_full.main()
    import sqlite3
    out = run / 'jobs' / args.method / args.dataset
    with sqlite3.connect(out / 'measurements.sqlite') as db:
        rows = [json.loads(r[0]) for r in db.execute('SELECT record FROM images ORDER BY ordinal')]
    expected = json.loads((run / 'expected_predictions.json').read_text())[args.method][args.dataset]
    different = [r['image_id'] for r in rows if r['raw_prediction_sha256'] != expected[r['image_id']]]
    report = dict(images=len(rows), exact_array_hash_matches=len(rows) - len(different), different_image_ids=different,
                  scope='Array bytes including dtype; a mismatch requires inspection, never silently substitutes published quality.')
    save(out / 'published_prediction_check.json', report)
    print(json.dumps(report))
    if different:
        raise RuntimeError('Predictions differ from published run')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--group', choices=['proposal', 'corrclip'], required=True)
    p.add_argument('--run', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--dataset', choices=['voc20', 'context59', 'ade150', 'coco_stuff171'], required=True)
    p.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else execute)(args)

if __name__ == '__main__':
    main()
