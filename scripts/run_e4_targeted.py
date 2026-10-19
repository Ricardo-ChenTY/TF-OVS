"""Only rerun the documented E4 configuration mismatch and online CorrCLIP variant."""
import argparse,json,os,shutil,subprocess,time,fcntl
from pathlib import Path
import numpy as np
from PIL import Image
from e4_full_common import ROOT,sha,save,utc,load_rows
from run_e4_full import FILES,summarize
SAM=['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']

def prepare(root):
    if (root/'plan.json').exists():raise RuntimeError('Refusing to overwrite frozen targeted plan')
    old=ROOT/'runs/rebuttal/e4_full_20261009';full=json.loads((old/'protocol.json').read_text())
    pilot=json.loads((ROOT/'runs/rebuttal/e4_20261009/protocol.json').read_text())
    jobs=json.loads((ROOT/'runs/rebuttal/execution_20261007/jobs.json').read_text())
    for group,methods in [('proposal_same_recipe',SAM),('corrclip_online',['corrclip_online'])]:
        run=root/group;(run/'code').mkdir(parents=True,exist_ok=True);(run/'manifests').mkdir(exist_ok=True)
        for name in FILES+['run_e4_targeted.py']:shutil.copy2(ROOT/'scripts'/name,run/'code'/name)
        protocol=json.loads(json.dumps(full));protocol.update(methods=methods,cudnn_tf32=group=='proposal_same_recipe',sampling=group=='proposal_same_recipe',
            measurement_scope='fixed_100_image_timing_with_original_cudnn_tf32' if group=='proposal_same_recipe' else 'full_validation_online_SAM2_8_variant',
            created_utc=utc(),tf32='matmul false; cudnn '+str(group=='proposal_same_recipe'),
            warmup_policy='5 per dataset; excluded from 25 measured samples for proposal; included in full pass for CorrCLIP',
            quality='proposal subset quality is diagnostic only; compare each prediction to archived full-quality run; CorrCLIP quality from own full timed predictions')
        references={}
        for ds,spec in protocol['datasets'].items():
            fullrows=load_rows(old/spec['manifest']);by_id={r['image_id']:r for r in fullrows}
            if group=='proposal_same_recipe':
                rows=[by_id[r['image_id']] for r in pilot['samples'][ds]['measure']]
                warm=[by_id[r['image_id']] for r in pilot['samples'][ds]['warmup']]
                wp=run/'manifests'/(ds+'_warmup.jsonl');wp.write_text(''.join(json.dumps(r)+'\n' for r in warm))
                spec.update(warmup_manifest=str(wp.relative_to(run)),warmup_sha256=sha(wp))
                for method in methods:
                    job=next(j for j in jobs if j['method']==method and j['dataset']==ds and j['seed']==0)
                    folder=Path(job['out'])/('candidates_'+str(job['sizes'][0]))
                    index={r['image_id']:r for r in load_rows(folder/'predictions.jsonl')}
                    references[method+'/'+ds]={r['image_id']:dict(path=str(folder/index[r['image_id']]['mask_path']),sha256=sha(folder/index[r['image_id']]['mask_path'])) for r in rows}
            else:rows=fullrows
            target=run/spec['manifest'];target.write_text(''.join(json.dumps(r)+'\n' for r in rows));spec.update(images=len(rows),sha256=sha(target))
        protocol.update(images_per_method=sum(s['images'] for s in protocol['datasets'].values()),total_jobs=4*len(methods))
        protocol['total_timed_predictions']=protocol['images_per_method']*len(methods)
        if references:
            save(run/'references.json',references);protocol['references_sha256']=sha(run/'references.json')
        sources={str(p):sha(p) for p in (run/'code').iterdir() if p.is_file()}
        for path,digest in json.loads((old/'source_sha256.json').read_text()).items():
            if str(old.resolve()/'code') in path:continue
            if sha(path)!=digest:raise RuntimeError('Original source changed: '+path)
            sources[path]=digest
        save(run/'source_sha256.json',sources);save(run/'protocol.json',protocol);summarize(run)
    save(root/'plan.json',dict(created_utc=utc(),reviewer='Z3',groups=['proposal_same_recipe','corrclip_online'],
        timed_predictions=13954,jobs=20,stopped_full_queue=str(old),
        rationale='Only restore 4 SAM recipes for 100-image timing and generate online SAM2-8 variant own full quality; no other full reruns',
        pairing_limit='Prediction equality on the fixed subset is a consistency check, not proof for untested images. Do not pool TF32-on timing with TF32-off uniform ablation without disclosure.',
        retained='Original 40 tasks, 18-row sampled E4 and 11 completed full method/dataset jobs remain intact'))

def audit_pair(run,method,ds):
    protocol=json.loads((run/'protocol.json').read_text())
    if sha(run/'references.json')!=protocol['references_sha256']:raise RuntimeError('Reference manifest changed')
    refs=json.loads((run/'references.json').read_text())[method+'/'+ds]
    out=run/'jobs'/method/ds;records=load_rows(out/'per_image.jsonl');rows=[]
    for r in records:
        ref=refs[r['image_id']]
        if sha(ref['path'])!=ref['sha256']:raise RuntimeError('Reference prediction changed')
        pred_path=out/r['prediction_file']
        if sha(pred_path)!=r['prediction_file_sha256']:raise RuntimeError('New prediction changed')
        pred=np.asarray(Image.open(pred_path));original=np.asarray(Image.open(ref['path']))
        if original.shape!=pred.shape:raise RuntimeError('Prediction geometry differs')
        rows.append(dict(image_id=r['image_id'],changed_pixels=int((original!=pred).sum()),pixels=int(pred.size)))
    report=dict(method=method,dataset=ds,compared=len(rows),exact_matches=sum(r['changed_pixels']==0 for r in rows),
                changed_pixels=sum(r['changed_pixels'] for r in rows),pixels=sum(r['pixels'] for r in rows),rows=rows,
                status='exact_on_measured_subset' if all(r['changed_pixels']==0 for r in rows) else 'mismatches_require_review',
                scope='only measured subset; full original scores remain their own results')
    save(out/'original_prediction_check.json',report);return report

def update(root):
    plan=json.loads((root/'plan.json').read_text());groups={g:summarize(root/g) for g in plan['groups']}
    paired=[]
    for path in (root/'proposal_same_recipe/jobs').glob('*/*/original_prediction_check.json'):
        r=json.loads(path.read_text());paired.append({k:r[k] for k in ['method','dataset','compared','exact_matches','status']})
    state=dict(updated_utc=utc(),groups=groups,completed_jobs=sum(g['completed_jobs'] for g in groups.values()),total_jobs=20,
               processed=sum(g['processed'] for g in groups.values()),expected=13954,paired_checks=paired,
               status='complete' if all(g['status']=='complete' for g in groups.values()) and len(paired)==16 else 'incomplete')
    save(root/'progress.json',state);return state

def queue(root):
    lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    failures=[]
    for group in ['proposal_same_recipe','corrclip_online']:
        run=root/group;protocol=json.loads((run/'protocol.json').read_text())
        for method in protocol['methods']:
            for ds in protocol['datasets']:
                out=run/'jobs'/method/ds;out.mkdir(parents=True,exist_ok=True)
                log=out/('worker_'+time.strftime('%Y%m%dT%H%M%S')+'.log')
                command=['bash',str(run/'code/python_e4'),str(run/'code/measure_e4_full.py'),'--run',str(run),'--method',method,'--dataset',ds]
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env.pop('E4_PYTHON',None)
                with log.open('w') as f:
                    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
                    save(root/'queue_status.json',dict(status='running',pid=os.getpid(),worker_pid=child.pid,group=group,method=method,dataset=ds,updated_utc=utc()))
                    while child.poll() is None:update(root);time.sleep(10)
                if child.returncode or not (out/'result.json').exists():
                    failures.append([group,method,ds]);save(root/'failures.json',failures);continue
                if group=='proposal_same_recipe':audit_pair(run,method,ds)
                print(group,method,ds,'complete',flush=True);update(root)
    result=update(root);save(root/'queue_status.json',dict(status=result['status'],failures=failures,updated_utc=utc()))
    matches=sum(r['exact_matches'] for r in result['paired_checks']);compared=sum(r['compared'] for r in result['paired_checks'])
    save(root/'verification.json',dict(status=result['status'],completed_jobs=result['completed_jobs'],paired_images=compared,exact_matches=matches,
         mismatch_requires_review=matches!=compared,full_quality_for_online_variant='corrclip_online/latency_by_dataset.csv',
         proposal_subset_quality_NOT_full_score='proposal_same_recipe/latency_by_dataset.csv',
         limitation='Old and restored timings use different cuDNN TF32 recipes; distinguish protocol-strict timing from native-recipe paired timing.'))
    (root/'summary_zh.txt').write_text('定向 E4 补测状态：'+result['status']+'\n完成任务 %s/20；旧预测逐张一致 %s/%s。\n'%(result['completed_jobs'],matches,compared)+
         '四种 SAM 方法为恢复旧 cuDNN TF32 设置的100张抽样计时；子集 mIoU 不替代原全量精度。\nCorrCLIP 在线 SAM2-8 全量分数属于该变体自身，不能替代缓存版本分数。\n严格统一 TF32-off 的原抽样表与 TF32-on 的旧配置配对表分开报告；本轮不自动宣称全方法 Pareto 排名。\n')
    if failures:raise RuntimeError(str(failures))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
    if a.action=='prepare':prepare(root);update(root)
    elif a.action=='run':queue(root)
    else:print(json.dumps(update(root),indent=2))
if __name__=='__main__':main()
