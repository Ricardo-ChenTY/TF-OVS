"""Persistence and metrics shared by the full-validation E4 runner."""
import hashlib, json, os, sqlite3, time, zlib
from pathlib import Path
import numpy as np

ROOT = Path('/data/tianyi/TF-OVOS')
DATASETS = {'voc20': ('voc20_val', 20, 1449), 'context59': ('context59_val', 59, 5105),
            'ade150': ('ade20k150_val', 150, 2000), 'coco_stuff171': ('coco_stuff171_val', 171, 5000)}
METHODS = ['maskclip_attn', 'sclip', 'proxyclip', 'scclip', 'trident', 'maskclip',
           'maskclip_attn_slide', 'cliptrase', 'naclip', 'resclip', 'corrclip',
           'corrclip_online', 'freeda', 'cass', 'sam_amg_clip', 'sam_amg_siglip',
           'dinov2_sam_clip', 'dinov2_sam_siglip']

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')
    os.replace(temp, path)

def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def open_db(path, identity):
    db = sqlite3.connect(str(path), timeout=60)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS images (image_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, record TEXT NOT NULL, confusion BLOB NOT NULL)')
    encoded = json.dumps(identity, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if old and old[0] != encoded:
        raise RuntimeError('Resume identity differs: protocol/code/manifest/method/dataset changed')
    with db:
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (encoded,))
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise RuntimeError('SQLite integrity check failed')
    return db

def confusion_matrix(gt, pred, classes):
    gt = np.asarray(gt); pred = np.asarray(pred)
    if gt.shape != pred.shape or pred.ndim != 2:
        raise ValueError('Prediction/GT geometry mismatch: %s versus %s' % (pred.shape, gt.shape))
    if not np.isfinite(pred).all() or not np.equal(pred, np.floor(pred)).all():
        raise ValueError('Expected integer label predictions')
    valid = (gt >= 0) & (gt < classes)
    if np.any(~valid & (gt != 255)):
        raise ValueError('Unexpected ground-truth labels')
    labels = pred[valid].astype(np.int64)
    # Predictions outside the evaluated vocabulary count as false negatives, never disappear.
    labels[(labels < 0) | (labels >= classes)] = classes
    return np.bincount(gt[valid].astype(np.int64) * (classes + 1) + labels,
                       minlength=classes * (classes + 1)).reshape(classes, classes + 1)

def pack_confusion(cm):
    return zlib.compress(np.asarray(cm, dtype='<i8').tobytes())

def unpack_confusion(blob, classes):
    return np.frombuffer(zlib.decompress(blob), dtype='<i8').reshape(classes, classes + 1)

def quality(cm):
    classes = cm.shape[0]; tp = np.diag(cm[:, :classes]).astype(float)
    gt = cm.sum(axis=1); pred = cm[:, :classes].sum(axis=0); union = gt + pred - tp
    iou = np.divide(tp, union, out=np.full(classes, np.nan), where=union > 0)
    return dict(miou_percent=float(np.nanmean(iou) * 100),
                per_class_iou_percent=[None if np.isnan(x) else float(x * 100) for x in iou],
                valid_pixels=int(gt.sum()), rejected_prediction_pixels=int(cm[:, -1].sum()),
                class_denominator='classes with nonzero union; ignore GT 255; invalid predictions count as FN')

def validate_prefix(db, rows, output):
    committed = list(db.execute('SELECT image_id, ordinal, record FROM images ORDER BY ordinal'))
    for index, (image_id, ordinal, record) in enumerate(committed):
        if ordinal != index or image_id != rows[index]['image_id']:
            raise RuntimeError('Committed records are not the exact manifest prefix')
        rec = json.loads(record); file = output / rec['prediction_file']
        if not file.is_file() or sha(file) != rec['prediction_file_sha256']:
            raise RuntimeError('A committed prediction is missing/corrupt: ' + str(file))
    return len(committed)

def finish_job(db, rows, output, metadata):
    if db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != len(rows):
        raise RuntimeError('Cannot finalize incomplete full-validation job')
    classes = metadata['classes']; cm = np.zeros((classes, classes + 1), dtype=np.int64); records = []
    for image_id, ordinal, record, blob in db.execute('SELECT * FROM images ORDER BY ordinal'):
        assert image_id == rows[ordinal]['image_id']
        records.append(json.loads(record)); cm += unpack_confusion(blob, classes)
    online = np.array([r['online_seconds'] for r in records]); pipeline = np.array([r['read_to_labels_seconds'] for r in records])
    result = dict(metadata, status='complete', completed_utc=utc(), images=len(records),
                  online_mean_seconds=float(online.mean()), online_total_seconds=float(online.sum()),
                  online_median_seconds=float(np.median(online)), online_p90_seconds=float(np.percentile(online, 90)),
                  online_p95_seconds=float(np.percentile(online, 95)), online_p99_seconds=float(np.percentile(online, 99)),
                  online_max_seconds=float(online.max()), read_to_labels_mean_seconds=float(pipeline.mean()),
                  peak_allocated_gib=max(r['peak_allocated_mib'] for r in records)/1024,
                  peak_reserved_gib=max(r['peak_reserved_mib'] for r in records)/1024, **quality(cm))
    np.save(output/'confusion.npy', cm)
    with (output/'per_image.jsonl').open('w') as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    save(output/'result.json', result)
    return result
