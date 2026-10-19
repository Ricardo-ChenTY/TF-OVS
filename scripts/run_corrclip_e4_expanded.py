"""Companion online CorrCLIP timing on exactly the proposal sample; serial GPU use."""
import argparse,csv,fcntl,json,os,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from e4_full_common import ROOT,save,sha,utc,load_rows
from run_e4_full import summarize
from run_proposal_e4_expanded import worker
from summarize_proposal_e4_expanded import bootstrap,interval,write

TARGETS=['voc20','context59','ade150','coco_stuff171']
HIST=ROOT/'runs/rebuttal/e4_targeted_20261009/corrclip_online'

def prepare(run,primary):
    if (run/'protocol.json').exists():raise RuntimeError('Already frozen')
    run.mkdir(parents=True,exist_ok=True)
    shutil.copytree(primary/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),run/'code'/Path(__file__).name)
    shutil.copytree(primary/'manifests',run/'manifests')
    protocol=json.loads((primary/'protocol.json').read_text())
    protocol.update(methods=['corrclip_online'],total_jobs=4,total_timed_predictions=4000,
        measurement_scope='online_CorrCLIP_SAM2_8_same_1000_images_per_target_as_proposals',
        cudnn_tf32=False,tf32='matmul off; cuDNN off, matching existing online CorrCLIP quality run',
        created_utc=utc(),primary_run=str(primary),primary_protocol_sha256=sha(primary/'protocol.json'),
        geometry='original frame; native online CorrCLIP SAM2-8 variant',
        prior_full_e4_not_reused='Remeasure same selected images; prior full predictions used only for consistency checks',
        quality='Compare against this online variant own full predictions; retain its existing full-run mean 48.35491454982702, never cached CorrCLIP quality')
    sources={str(p):sha(p) for p in (run/'code').rglob('*') if p.is_file()}
    old=json.loads((HIST/'source_sha256.json').read_text())
    for path,digest in old.items():
        if '/CorrCLIP/' in path:
            assert sha(path)==digest,('Historical CorrCLIP source changed',path)
            sources[path]=digest
    cache=Path('/data/sdd/tianyi/TF-OVOS-rebuttal/.cache/e4')
    for p in [cache/'weights/sam2_hiera_large.pt',cache/'clip/b16_fullcc2.5b.pt',cache/'torch/hub/checkpoints/dino_vitbase8_pretrain.pth']:
        sources[str(p)]=sha(p)
    refs={}
    for ds,spec in protocol['datasets'].items():
        assert sha(run/spec['manifest'])==spec['sha256']
        base=HIST/'jobs/corrclip_online'/ds
        meta=json.loads((base/'metadata.json').read_text())['settings']
        repo=Path(meta['repo']);cfg=repo/'configs'/('cfg_tfovos_'+{'voc20':'voc20','context59':'context59','ade150':'ade20k','coco_stuff171':'coco_stuff164k'}[ds]+'.py')
        assert sha(cfg)==meta['config_sha256']
        vocab=repo/meta['model']['name_path'];assert sha(vocab)==meta['class_vocabulary_sha256']
        sources[str(cfg)]=sha(cfg);sources[str(vocab)]=sha(vocab)
        index={r['image_id']:r for r in load_rows(base/'per_image.jsonl')}
        refs[ds]={}
        for r in load_rows(run/spec['manifest']):
            saved=index[r['image_id']];p=base/saved['prediction_file']
            assert sha(p)==saved['prediction_file_sha256']
            refs[ds][r['image_id']]=dict(path=str(p),sha256=sha(p))
    save(run/'references.json',refs);protocol['references_sha256']=sha(run/'references.json')
    save(run/'source_sha256.json',sources);save(run/'protocol.json',protocol);summarize(run)

def audit(run,ds,limit=None):
    import sqlite3
    protocol=json.loads((run/'protocol.json').read_text());assert sha(run/'references.json')==protocol['references_sha256']
    refs=json.loads((run/'references.json').read_text())[ds];out=run/'jobs/corrclip_online'/ds
    db=sqlite3.connect(str(out/'measurements.sqlite'));records=[json.loads(r[0]) for r in db.execute('SELECT record FROM images ORDER BY ordinal')];db.close()
    if limit is not None:records=records[:limit]
    results=[]
    for r in records:
        p=out/r['prediction_file'];ref=Path(refs[r['image_id']]['path'])
        assert sha(p)==r['prediction_file_sha256'] and sha(ref)==refs[r['image_id']]['sha256']
        a=np.asarray(Image.open(p));b=np.asarray(Image.open(ref));assert a.shape==b.shape
        results.append(dict(image_id=r['image_id'],changed_pixels=int((a!=b).sum()),pixels=a.size,prediction_sha256=sha(p),reference_sha256=sha(ref)))
    result=dict(images=len(results),exact_matches=sum(r['changed_pixels']==0 for r in results),records=results)
    save(out/('smoke_pairing.json' if limit else 'prediction_pairing.json'),result)
    assert result['images']==result['exact_matches']==(limit or 1000),'Prediction mismatch; do not pair with old quality'

def intervals(run,primary):
    arrays=[];rows=[];sources={}
    for ds in TARGETS:
        out=run/'jobs/corrclip_online'/ds;rs=load_rows(out/'per_image.jsonl')
        assert [r['image_id'] for r in rs]==[r['image_id'] for r in load_rows(primary/'manifests'/(ds+'.jsonl'))]
        check=json.loads((out/'prediction_pairing.json').read_text());assert check['images']==check['exact_matches']==1000
        a=np.array([r['online_seconds'] for r in rs]);assert len(a)==1000 and np.isfinite(a).all() and (a>0).all()
        result=json.loads((out/'result.json').read_text());assert np.isclose(a.mean(),result['online_mean_seconds'],atol=1e-12,rtol=0)
        arrays.append(a)
        for name in ['per_image.jsonl','result.json','prediction_pairing.json']:sources[str((out/name).relative_to(run))]=sha(out/name)
    per,macro=bootstrap(arrays)
    for ds,a,b in list(zip(TARGETS,arrays,per))+[('equal_target_mean',np.concatenate(arrays),macro)]:
        lo,hi=interval(b);rows.append(dict(method='corrclip_online_sam2_8',target=ds,images=len(a),online_mean_seconds=float(a.mean()),ci95_low_seconds=lo,ci95_high_seconds=hi))
    write(run/'latency_ci.csv',rows)
    with (primary/'latency_ci.csv').open() as f:combined=list(csv.DictReader(f))
    for row in combined:row['cudnn_tf32']=True
    for row in rows:row['cudnn_tf32']=False
    write(run/'latency_ci_five_configurations.csv',combined+rows)
    save(run/'interval_verification.json',dict(status='complete',same_image_ids_per_target=1000,total_matching_predictions_across_five_configs=20000,
        primary_intervals_sha256=sha(primary/'latency_ci.csv'),sources=sources,
        scope='Same 4000 images, equal target weights, same timer boundaries and hardware; native recipes retained including cuDNN TF32 difference. Image-bootstrap uncertainty only.',
        quality='Use previous full-run quality for each verified configuration. Sample quality is not a replacement.'))

def queue(run,primary):
    lock=(run/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        protocol=json.loads((run/'protocol.json').read_text());assert sha(primary/'protocol.json')==protocol['primary_protocol_sha256']
        while True:
            state=json.loads((primary/'queue_status.json').read_text())
            if state['status']=='failed':raise RuntimeError('Proposal queue failed; companion not launched')
            if state['status']=='complete':break
            save(run/'queue_status.json',dict(status='waiting_for_proposal_queue',primary_status=state['status'],updated_utc=utc(),queue_pid=os.getpid()))
            time.sleep(30)
        # The primary sets complete only after its last GPU worker exits and statistics pass.
        worker(run,'corrclip_online','voc20',2);audit(run,'voc20',2)
        for ds in TARGETS:
            worker(run,'corrclip_online',ds);audit(run,ds)
        intervals(run,primary);summarize(run)
        save(run/'queue_status.json',dict(status='complete',updated_utc=utc()))
    except BaseException as exc:
        save(run/'queue_status.json',dict(status='failed',error=repr(exc),updated_utc=utc()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--run',type=Path,required=True);p.add_argument('--primary',type=Path,required=True);a=p.parse_args()
    if a.action=='prepare':prepare(a.run.resolve(),a.primary.resolve())
    else:queue(a.run.resolve(),a.primary.resolve())
if __name__=='__main__':main()
