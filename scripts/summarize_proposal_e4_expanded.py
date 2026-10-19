"""Stratified image-bootstrap intervals for expanded E4 timings; no GPU."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from e4_full_common import save,sha,load_rows
METHODS=['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']
TARGETS=['voc20','context59','ade150','coco_stuff171']

def interval(values):
    return np.percentile(values,[2.5,97.5]).tolist()

def bootstrap(arrays,draws=10000,seed=20261015):
    """Each array is a target's aligned image measurements (or paired differences)."""
    rng=np.random.default_rng(seed)
    boots=[a[rng.integers(0,len(a),size=(draws,len(a)))].mean(1) for a in arrays]
    return boots,np.mean(boots,axis=0)

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize_intervals(run):
    run=Path(run);data={};sources={};table=[]
    for method in METHODS:
        for ds in TARGETS:
            out=run/'jobs'/method/ds
            result=json.loads((out/'result.json').read_text());audit=json.loads((out/'prediction_pairing.json').read_text())
            assert result['images']==audit['images']==audit['exact_matches']==1000
            assert result['timing_scope']=='online_after_class_setup'
            rows=load_rows(out/'per_image.jsonl');manifest=load_rows(run/'manifests'/(ds+'.jsonl'))
            assert [r['image_id'] for r in rows]==[r['image_id'] for r in manifest]
            a=np.array([r['online_seconds'] for r in rows]);assert len(a)==1000 and np.all(a>0) and np.isfinite(a).all()
            assert np.isclose(a.mean(),result['online_mean_seconds'],rtol=0,atol=1e-12)
            data[method,ds]=a
            for name in ['per_image.jsonl','result.json','prediction_pairing.json']:
                sources[str((out/name).relative_to(run))]=sha(out/name)
        arrays=[data[method,ds] for ds in TARGETS];per,macro=bootstrap(arrays)
        for ds,a,b in zip(TARGETS,arrays,per):
            lo,hi=interval(b);table.append(dict(method=method,target=ds,images=1000,online_mean_seconds=float(a.mean()),ci95_low_seconds=lo,ci95_high_seconds=hi))
        lo,hi=interval(macro);table.append(dict(method=method,target='equal_target_mean',images=4000,online_mean_seconds=float(np.mean([a.mean() for a in arrays])),ci95_low_seconds=lo,ci95_high_seconds=hi))
    differences=[]
    for suffix in ['clip','siglip']:
        arrays=[data['dinov2_sam_'+suffix,ds]-data['sam_amg_'+suffix,ds] for ds in TARGETS]
        per,macro=bootstrap(arrays)
        for ds,a,b in list(zip(TARGETS,arrays,per))+[('equal_target_mean',np.concatenate(arrays),macro)]:
            lo,hi=interval(b);differences.append(dict(naming=suffix,target=ds,paired_images=len(a),dino_minus_sam_mean_seconds=float(a.mean()),ci95_low_seconds=lo,ci95_high_seconds=hi))
    write(run/'latency_ci.csv',table);write(run/'paired_dino_overhead_ci.csv',differences)
    save(run/'interval_verification.json',dict(status='complete',verified_predictions=16000,draws=10000,seed=20261015,source_sha256=sources,
        scope='Percentile bootstrap over image IDs within each target; equal target weights; paired image differences for DINO overhead',
        limitations='Conditional on these sessions. Does not measure between-run, thermal, order, or hardware variability; no proof of a universal 0.05-second overhead.',
        quality='Retain original full-quality scores. The sample mIoUs are diagnostics only.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();summarize_intervals(a.run)
