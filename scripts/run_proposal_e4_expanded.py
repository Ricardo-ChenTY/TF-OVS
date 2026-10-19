"""Time four current proposal configurations on 1000 shared images per target."""
import argparse
import fcntl
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import time
from e4_full_common import ROOT, save, sha, utc, load_rows
from run_e4_full import summarize
METHODS = ['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']
FILES = ['run_proposal_e4_expanded.py','summarize_proposal_e4_expanded.py','measure_proposal_e4_full.py','run_e4_full.py',
         'measure_e4_full.py','e4_full_common.py','e4_gpu_guard.py','measure_e4.py',
         'e4_special_backends.py','python_e4']

def prepare(run):
    if (run/'protocol.json').exists() or (run/'CANCELLED.json').exists():raise RuntimeError('Already frozen or cancelled')
    (run/'code').mkdir(parents=True,exist_ok=True)
    (run/'manifests').mkdir(exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'scripts'/name,run/'code'/name)
    shutil.copytree(ROOT/'src/tf_ovos',run/'code/src/tf_ovos',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    old=ROOT/'runs/rebuttal/e4_full_20261009'
    protocol=json.loads((old/'protocol.json').read_text())
    for ds,spec in protocol['datasets'].items():
        rows=load_rows(old/spec['manifest'])
        assert len(rows)==spec['images']
        warm=[r for r in rows if r['image_id'] in spec['warmup_ids']]
        assert len(warm)==5
        candidates=sorted((r for r in rows if r['image_id'] not in spec['warmup_ids']),key=lambda r:r['image_id'])
        rows=random.Random(20261015).sample(candidates,1000)
        for row in rows+warm:
            for key in ['image','mask']:
                row[key+'_sha256']=sha(row[key+'_path'])
        dest=run/spec['manifest'];dest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        wp=run/'manifests'/(ds+'_warmup.jsonl');wp.write_text(''.join(json.dumps(r)+'\n' for r in warm))
        spec.update(images=1000,sha256=sha(dest),warmup_manifest=str(wp.relative_to(run)),warmup_sha256=sha(wp))
        print('Frozen inputs',ds,len(rows),flush=True)
    protocol.update(methods=METHODS,total_jobs=16,total_timed_predictions=16000,images_per_method=4000,
        created_utc=utc(),sampling=True,sampling_seed=20261015,samples_per_target=1000,cudnn_tf32=True,tf32='matmul off; cuDNN on',
        measurement_scope='expanded_1000_per_target_timing_current_corrected_proposal_recipes',
        warmup_policy='5 fixed images per target per process; excluded from measured sample',
        sampling_policy='uniform without replacement from sorted full manifest excluding 5 warmups; identical IDs and order for all methods',
        confidence_intervals='10000 stratified image bootstrap draws, seed 20261015; percentile 95%; paired images for DINO-minus-SAM contrasts; conditional on measured sessions, not repeated-run uncertainty',
        quality='sample mIoU diagnostic only; pixelwise agreement with archived full-quality predictions required; never replace Table 1 quality',
        cache_policy='no prediction reuse or visual-stage caching; text features initialized before online timing',
        geometry='original frame; corrected DINO token-mask geometry; SigLIP eval preprocessing',
        environment='Torch 2.12; fixed four-thread runtime; frozen current adapter package',
        aggregation='four-target equal-weight latency primary; report 95% stratified image-bootstrap interval; retain original full-quality values',
        prior_full_e4_not_reused='Old proposal jobs have no full results; old global protocol used cuDNN TF32 off')
    sources={str(p):sha(p) for p in (run/'code').rglob('*') if p.is_file()}
    for name in ['voc20.txt','context_59.txt','ade20k_150.txt','coco_stuff_171.txt']:
        p=ROOT/'configs/vocab'/name;sources[str(p)]=sha(p)
    # Freeze local model artifacts too; startup detects any subsequent replacement.
    paths=[ROOT/'weights/sam_vit_h_4b8939.pth']
    for cache in ['clip','siglip','dinov2']:
        paths.extend(p for p in (ROOT/'.cache'/cache).rglob('*') if p.is_file() and '/.locks/' not in str(p))
    for p in paths:sources[str(p)]=sha(p)
    refs=freeze_references(run)
    save(run/'references.json',refs);protocol['references_sha256']=sha(run/'references.json')
    save(run/'source_sha256.json',sources);save(run/'protocol.json',protocol)
    summarize(run)

def freeze_references(run):
    refs={}
    jobs=json.loads((ROOT/'runs/rebuttal/execution_20261007/jobs.json').read_text())
    for method in METHODS:
        for ds in ['voc20','context59','ade150','coco_stuff171']:
            rows=load_rows(run/'manifests'/(ds+'.jsonl'))
            job=next(j for j in jobs if j['method']==method and j['dataset']==ds and j['seed']==0)
            if method.startswith('dinov2_'):
                ids={r['image_id']:i for i,r in enumerate(load_rows(Path(job['out'])/'manifest.jsonl'))}
                base=ROOT/'runs/rebuttal/closure_20261011/dino_geometry/full'/job['name']/('candidates_'+str(job['sizes'][0]))
                paths={r['image_id']:base/('%06d.png'%ids[r['image_id']]) for r in rows}
            else:
                base=Path(job['out'])/('candidates_'+str(job['sizes'][0]))
                idx={r['image_id']:r for r in load_rows(base/'predictions.jsonl')}
                paths={r['image_id']:base/idx[r['image_id']]['mask_path'] for r in rows}
            refs[method+'/'+ds]={key:dict(path=str(path),sha256=sha(path)) for key,path in paths.items()}
    return refs

def audit(run,method,ds,limit=None):
    import sqlite3
    import numpy as np
    from PIL import Image
    out=run/'jobs'/method/ds
    db=sqlite3.connect(str(out/'measurements.sqlite'))
    records=[json.loads(r[0]) for r in db.execute('SELECT record FROM images ORDER BY ordinal')];db.close()
    if limit is not None:records=records[:limit]
    protocol=json.loads((run/'protocol.json').read_text())
    assert sha(run/'references.json')==protocol['references_sha256']
    refs=json.loads((run/'references.json').read_text())[method+'/'+ds]
    checks=[]
    for r in records:
        current=out/r['prediction_file'];ref=Path(refs[r['image_id']]['path'])
        assert sha(ref)==refs[r['image_id']]['sha256']
        assert sha(current)==r['prediction_file_sha256']
        a=np.asarray(Image.open(current));b=np.asarray(Image.open(ref));assert a.shape==b.shape
        checks.append(dict(image_id=r['image_id'],changed_pixels=int((a!=b).sum()),pixels=a.size,
                           prediction_sha256=sha(current),reference_sha256=sha(ref)))
    result=dict(images=len(checks),exact_matches=sum(r['changed_pixels']==0 for r in checks),
                changed_pixels=sum(r['changed_pixels'] for r in checks),records=checks)
    save(out/('smoke_pairing.json' if limit else 'prediction_pairing.json'),result)
    return result

def worker(run,method,ds,stop=None):
    out=run/'jobs'/method/ds;out.mkdir(parents=True,exist_ok=True)
    command=['bash',str(run/'code/python_e4'),str(run/'code/measure_proposal_e4_full.py'),
             '--run',str(run),'--method',method,'--dataset',ds]
    if stop:command+=['--stop-after',str(stop)]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env.pop('E4_PYTHON',None)
    logfile=out/('worker_'+time.strftime('%Y%m%dT%H%M%S')+'.log')
    with logfile.open('w') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        save(run/'queue_status.json',dict(status='smoke' if stop else 'running',method=method,dataset=ds,
            worker_pid=child.pid,queue_pid=os.getpid(),log=str(logfile),updated_utc=utc()))
        while child.poll() is None:summarize(run);time.sleep(10)
    if child.returncode:raise RuntimeError('Worker failed: '+str(logfile))
    summarize(run)

def queue(run):
    if (run/'CANCELLED.json').exists():raise RuntimeError('Cancelled run')
    lock=(run/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # Verify all four adapters on the first two selected images, then resume them.
        for method in METHODS:
            smoke=run/'jobs'/method/'voc20/smoke_pairing.json'
            if smoke.exists():
                r=json.loads(smoke.read_text())
            else:
                worker(run,method,'voc20',2);r=audit(run,method,'voc20',2)
            if r['images']!=2 or r['exact_matches']!=2:raise RuntimeError('Smoke predictions differ: '+method)
        save(run/'smoke_complete.json',dict(status='passed',images=8,methods=METHODS,updated_utc=utc()))
        for ds in json.loads((run/'protocol.json').read_text())['datasets']:
            for method in ['sam_amg_clip','dinov2_sam_clip','sam_amg_siglip','dinov2_sam_siglip']:
                worker(run,method,ds)
                r=audit(run,method,ds)
                print('Completed',method,ds,'exact',r['exact_matches'],'/',r['images'],flush=True)
                if r['images']!=1000 or r['exact_matches']!=1000:raise RuntimeError('Prediction mismatch; investigate before publishing timing/quality pair')
        result=summarize(run)
        from summarize_proposal_e4_expanded import summarize_intervals
        summarize_intervals(run)
        save(run/'queue_status.json',dict(status=result['status'],updated_utc=utc()))
    except BaseException as exc:
        save(run/'queue_status.json',dict(status='failed',error=repr(exc),updated_utc=utc()))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status']);p.add_argument('--run',required=True,type=Path)
    a=p.parse_args();run=a.run.resolve()
    if a.action=='prepare':prepare(run)
    elif a.action=='run':queue(run)
    else:print(json.dumps(summarize(run),indent=2))
if __name__=='__main__':main()
