"""Unified seed-0 reporting; reuse completed proposal data without GPU reruns."""
import argparse, csv, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path('/data/tianyi/TF-OVOS')
PROPOSALS=['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']
SIZES={'ade150':[150,300,500,847],'context59':[59,150,300,459]}
IMAGES={'ade150':2000,'context59':5105}
CURVE_FIELDS=['method','dataset','candidate_seed','candidate_draws','images','base_candidates','candidates','miou_percent','geometry','source','source_sha256','vocabulary_sha256']
END_FIELDS=['method','dataset','candidate_seed','candidate_draws','images','base_candidates','max_candidates','base_miou_percent','expanded_miou_percent','drop_pp','geometry','base_source','expanded_source']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p)
def csv_save(p,rows,fields):
    temp=p.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,p)

def proposal_rows():
    archive=ROOT/'runs/rebuttal/final_analysis_20261009';hashes=read(archive/'verification.json')['source_sha256']
    with (archive/'all_completed_metrics.csv').open() as f:records=list(csv.DictReader(f))
    selected=[r for r in records if r['method'] in PROPOSALS and r['dataset'] in SIZES and r['seed']=='0']
    assert len(selected)==32
    result=[]
    for r in selected:
        p=Path(r['source']);digest=sha(p);assert hashes[str(p)]==digest
        d=read(p);ds=r['dataset'];n=int(r['candidates']);assert d['num_images']==IMAGES[ds] and d['num_classes']==SIZES[ds][0]
        assert d['num_candidates']==n and n in SIZES[ds] and d['sample_scope']=='full_manifest'
        assert abs(d['miou']*100-float(r['miou_percent']))<1e-12
        assert sha(p.parent/'vocabulary.txt')==d['vocab_sha256']
        result.append(dict(method=r['method'],dataset=ds,candidate_seed=0,candidate_draws=1,images=d['num_images'],
            base_candidates=d['num_classes'],candidates=n,miou_percent=d['miou']*100,geometry='original_frame',
            source=str(p),source_sha256=digest,vocabulary_sha256=d['vocab_sha256']))
    assert len({(r['method'],r['dataset'],r['candidates']) for r in result})==32
    return result

def endpoints(rows):
    result=[]
    for method,ds in dict.fromkeys((r['method'],r['dataset']) for r in rows):
        rr={r['candidates']:r for r in rows if r['method']==method and r['dataset']==ds}
        low,high=SIZES[ds][0],SIZES[ds][-1]
        if low not in rr or high not in rr:continue
        a,b=rr[low],rr[high];assert a['images']==b['images'] and a['geometry']==b['geometry']
        result.append(dict(method=method,dataset=ds,candidate_seed=0,candidate_draws=1,images=a['images'],
            base_candidates=low,max_candidates=high,base_miou_percent=a['miou_percent'],expanded_miou_percent=b['miou_percent'],
            drop_pp=a['miou_percent']-b['miou_percent'],geometry=a['geometry'],base_source=a['source'],expanded_source=b['source']))
    return result

def nonmonotonic(rows):
    result=[]
    for method,ds in dict.fromkeys((r['method'],r['dataset']) for r in rows):
        rr=sorted([r for r in rows if r['method']==method and r['dataset']==ds],key=lambda r:r['candidates'])
        for a,b in zip(rr,rr[1:]):
            if SIZES[ds].index(b['candidates'])-SIZES[ds].index(a['candidates'])!=1:continue
            if b['miou_percent']>a['miou_percent']+1e-10:
                result.append(dict(method=method,dataset=ds,candidate_seed=0,from_candidates=a['candidates'],to_candidates=b['candidates'],increase_pp=b['miou_percent']-a['miou_percent']))
    return result

def report(run,out,proposals):
    plan=read(run/'plan.json');rows=list(proposals)
    vocab={(r['dataset'],r['candidates']):r['vocabulary_sha256'] for r in proposals if r['method']==PROPOSALS[0]}
    for r in proposals:assert r['vocabulary_sha256']==vocab[(r['dataset'],r['candidates'])]
    sources={r['source']:r['source_sha256'] for r in proposals}
    for method in plan['new_methods']:
        for job in read(run/'methods'/method/'plan.json')['jobs']:
            if not (job['name'].endswith('_base') or '_s0_' in job['name']):continue
            folder=Path(job['run']);p=folder/'jobs'/method/job['dataset']/'result.json';check=p.parent/'independent_verification.json'
            if not p.exists() or not check.exists():continue
            d=read(p);v=read(check);protocol=read(folder/'protocol.json')
            assert d['status']=='complete' and d['images']==v['images']==IMAGES[job['dataset']]
            assert v['confusion_from_exported_predictions_matches'] and v['extra_predictions_count_as_FN']
            assert d['protocol_sha256']==sha(folder/'protocol.json') and d['source_manifest_sha256']==sha(folder/'source_sha256.json')
            h=sha(folder/'vocabulary.txt');assert h==protocol['candidate_vocabulary']['sha256']==vocab[(job['dataset'],job['candidates'])]
            digest=sha(p);sources[str(p)]=digest;sources[str(check)]=sha(check)
            rows.append(dict(method=method,dataset=job['dataset'],candidate_seed=0,candidate_draws=1,images=d['images'],
                base_candidates=job['base'],candidates=job['candidates'],miou_percent=d['miou_percent'],geometry=protocol['geometry'],
                source=str(p),source_sha256=digest,vocabulary_sha256=h))
    assert len({(r['method'],r['dataset'],r['candidates']) for r in rows})==len(rows)
    ends=endpoints(rows);coverage=[]
    for method in PROPOSALS+plan['new_methods']:
        for ds,sizes in SIZES.items():
            levels=[r['candidates'] for r in rows if r['method']==method and r['dataset']==ds]
            coverage.append(dict(method=method,dataset=ds,candidate_seed=0,completed_levels=len(levels),expected_levels=4,
                complete_curve=set(levels)==set(sizes),complete_endpoints=sizes[0] in levels and sizes[-1] in levels))
    csv_save(out/'curves.csv',rows,CURVE_FIELDS);csv_save(out/'endpoints.csv',ends,END_FIELDS)
    csv_save(out/'coverage.csv',coverage,list(coverage[0]))
    csv_save(out/'nonmonotonic_steps.csv',nonmonotonic(rows),['method','dataset','candidate_seed','from_candidates','to_candidates','increase_pp'])
    save(out/'verification.json',dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),candidate_seed=0,candidate_draws=1,
        methods_with_both_complete_curves=sum(all(c['complete_curve'] for c in coverage if c['method']==m) for m in PROPOSALS+plan['new_methods']),
        total_methods=17,proposal_metrics_reused=32,completed_metric_rows=len(rows),complete_endpoint_rows=len(ends),
        source_sha256=sources,reporter_sha256=sha(__file__),scope='Main fixed-GT report: seed 0 only, full images. No seed averaging or SD. Historical three-seed tables are supplementary.',
        validation='Proposal metrics checked against the archived verified hashes and seed0 raw scores; new metrics require full result and independent PNG verification. Candidate vocabulary hashes match across methods.',
        limits=['Single candidate draw does not estimate seed variability','CLIPtrase retains native center-crop geometry','Candidate aliases/semantic overlaps retained','No new model inference was performed for proposal seed selection']))
    return coverage

def plot_proposals(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['SAM+CLIP','SAM+SigLIP','DINOv2+SAM+CLIP','DINOv2+SAM+SigLIP']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,(ds,sizes) in zip(axes,SIZES.items()):
        for method,name in zip(PROPOSALS,names):
            rr=sorted([r for r in rows if r['dataset']==ds and r['method']==method],key=lambda r:r['candidates'])
            ax.plot([r['candidates'] for r in rr],[r['miou_percent'] for r in rr],marker='o',label=name)
        ax.set(title=ds+' (candidate seed 0)',xlabel='Candidate vocabulary size',ylabel='mIoU (%)',xticks=sizes);ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ['pdf','png']:fig.savefig(out/('proposal_seed0_expansion.'+ext),dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=60)
    a=p.parse_args();run=a.run.resolve();out=run/'seed0_report';out.mkdir(exist_ok=True)
    lock=(out/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proposals=proposal_rows();csv_save(out/'proposal_seed0_curves.csv',proposals,CURVE_FIELDS);csv_save(out/'proposal_seed0_endpoints.csv',endpoints(proposals),END_FIELDS)
    plot_proposals(out,proposals)
    while True:
        coverage=report(run,out,proposals)
        if not a.watch or all(c['complete_curve'] for c in coverage):break
        if subprocess.run(['systemctl','--user','is-active','--quiet','tf-ovos-all-fixed-gt-20261009.service']).returncode!=0:break
        time.sleep(a.interval)
    print('Seed-0 report refreshed:',out,flush=True)

if __name__=='__main__':main()
