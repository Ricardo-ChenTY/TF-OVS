"""Freeze, run serially, resume and summarize full-validation E4 comparisons."""
import argparse, csv, fcntl, json, os, shutil, subprocess, time
from pathlib import Path
from e4_full_common import ROOT, DATASETS, METHODS, utc, sha, save, load_rows

FILES = ['run_e4_full.py','measure_e4_full.py','e4_full_common.py','e4_gpu_guard.py',
         'measure_e4.py','e4_special_backends.py','python_e4']

def prepare(run):
    if (run/'protocol.json').exists():raise RuntimeError('Already frozen; refusing overwrite')
    run.mkdir(parents=True,exist_ok=True);(run/'code').mkdir(exist_ok=True);(run/'manifests').mkdir(exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'scripts'/name,run/'code'/name)
    pilot=json.loads((ROOT/'runs/rebuttal/e4_20261009/protocol.json').read_text())
    datasets={}
    for ds,(stem,classes,count) in DATASETS.items():
        source=ROOT/'data/manifests'/(stem+'.jsonl');rows=load_rows(source);assert len(rows)==count
        assert len({r['image_id'] for r in rows})==count
        frozen=[]
        for r in rows:
            item={'image_id':r['image_id']}
            for key in ['image_path','mask_path']:
                path=(source.parent/r[key]).resolve();item[key]=str(path);item[key.replace('_path','_sha256')]=sha(path)
            frozen.append(item)
        dest=run/'manifests'/(ds+'.jsonl')
        dest.write_text(''.join(json.dumps(r)+'\n' for r in frozen))
        datasets[ds]=dict(manifest=str(dest.relative_to(run)),sha256=sha(dest),images=count,classes=classes,
                          warmup_ids=[r['image_id'] for r in pilot['samples'][ds]['warmup']])
        assert set(datasets[ds]['warmup_ids'])<=set(r['image_id'] for r in frozen)
        print('FROZEN',ds,count,flush=True)
    sources={str(p):sha(p) for p in (run/'code').iterdir() if p.is_file()}
    old=json.loads((ROOT/'runs/rebuttal/e4_20261009/source_sha256.json').read_text())
    for path in old:
        if 'e4_20261009/code' in path:continue
        path=(ROOT/path).resolve();sources[str(path)]=sha(path)
    save(run/'source_sha256.json',sources)
    save(run/'protocol.json',dict(created_utc=utc(),methods=METHODS,datasets=datasets,sampling=False,
          images_per_method=sum(d['images'] for d in datasets.values()),total_jobs=len(METHODS)*len(datasets),
          total_timed_predictions=len(METHODS)*sum(d['images'] for d in datasets.values()),
          batch_size=1,warmup_policy='5 per dataset per process; warmup images ALSO timed in full pass',
          cuda_synchronize=True,tf32=False,cudnn_benchmark=False,threads=4,
          online_timer='after disk read/RGB decode through native preprocessing, inference and CPU labels; CUDA synchronized',
          read_to_labels_timer='includes input read/decode and cached-mask read where applicable; excludes output writes, GT and scoring',
          setup_timer='model load plus class text/prototype setup, separately per process; historical offline index cost unknown',
          cache_policy='no prediction or cross-method visual-feature reuse; native CorrCLIP cached-mask row explicitly supplemental',
          quality='mIoU computed from the exact timed predictions; canonical compact GT, void=255',
          geometry='native method recipes retained; CLIPtrase uses native 336 center crop and is not full-frame comparable',
          environment='FreeDA uses native Torch 1.12.1; others Torch 2.12; native resolutions and dtypes disclosed per job',
          aggregation='per-dataset numbers primary; macro mean over four dataset means and pooled per-image mean both exported',
          concurrency='one method/dataset process; other observed GPU users pause/reject measurement; no other process killed',
          contention_detection='NVML before/after each inference plus 250 ms polling; shorter transient processes may evade detection'))
    (run/'README.txt').write_text('E4 全量复测：18 行（17 个方法 + CorrCLIP 在线 SAM2-8 补充变体），72 个方法/数据集任务，243972 次计时预测。\n每行完整覆盖 VOC 1449、Context 5105、ADE 2000、COCO 5000；不抽样。\n计时与 mIoU 来自同一轮预测。原生分辨率/裁剪/精度和 FreeDA 环境差异均披露，不宣称所有配置完全相同。\nCorrCLIP 缓存掩码行不是端到端；在线 SAM2-8 变体独立报告自己的精度。CLIPtrase 为原生 336 中心裁剪。\n离线索引历史生成成本未知，不能按零处理。setup_sessions 单独报告初始化，不计入在线单图耗时。\nSQLite 逐图提交，预测文件 SHA256 校验后才能续跑；冻结代码、清单及输入/GT SHA256。\n运行：bash code/python_e4 code/run_e4_full.py run --run 本目录\n查看：bash code/python_e4 code/run_e4_full.py status --run 本目录\n完整结果自动生成 latency_by_dataset.csv、comparison.csv 和 summary_zh.txt；覆盖不足不会标记完成。\n')
    summarize(run)

def write_csv(path,rows,fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)

def summarize(run):
    protocol=json.loads((run/'protocol.json').read_text());coverage=[];results=[];groups={}
    protocol_hash=sha(run/'protocol.json');source_hash=sha(run/'source_sha256.json')
    for method in protocol['methods']:
        groups[method]=[]
        for ds,spec in protocol['datasets'].items():
            out=run/'jobs'/method/ds;result=out/'result.json';status=out/'status.json'
            state=json.loads(status.read_text()) if status.exists() else {}
            rec=dict(method=method,dataset=ds,processed=state.get('processed',0),expected=spec['images'],status=state.get('status','pending'))
            if result.exists():
                r=json.loads(result.read_text())
                if (r['images']!=spec['images'] or r['protocol_sha256']!=protocol_hash or r['source_manifest_sha256']!=source_hash
                    or r['manifest_sha256']!=spec['sha256']):raise RuntimeError('Invalid completed result: '+str(result))
                rec.update(processed=r['images'],status='complete');results.append(r);groups[method].append(r)
            coverage.append(rec)
    write_csv(run/'coverage.csv',coverage,['method','dataset','processed','expected','status'])
    fields=['method','dataset','images','online_mean_seconds','online_median_seconds','online_p95_seconds','online_p99_seconds',
            'read_to_labels_mean_seconds','peak_allocated_gib','peak_reserved_gib','miou_percent','timing_scope','quality_geometry','torch_version']
    write_csv(run/'latency_by_dataset.csv',results,fields)
    comparison=[]
    for method,rs in groups.items():
        if len(rs)!=len(protocol['datasets']):continue
        comparison.append(dict(method=method,images=sum(r['images'] for r in rs),
            macro_online_seconds=sum(r['online_mean_seconds'] for r in rs)/len(rs),
            pooled_online_seconds=sum(r['online_total_seconds'] for r in rs)/sum(r['images'] for r in rs),
            macro_miou_percent=sum(r['miou_percent'] for r in rs)/len(rs),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in rs),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rs),
            timing_scope=rs[0]['timing_scope'],quality_geometry=rs[0]['quality_geometry']))
    write_csv(run/'comparison.csv',comparison,['method','images','macro_online_seconds','pooled_online_seconds','macro_miou_percent',
              'peak_allocated_gib','peak_reserved_gib','timing_scope','quality_geometry'])
    summary=dict(updated_utc=utc(),completed_jobs=len(results),total_jobs=protocol['total_jobs'],
                 processed=sum(r['processed'] for r in coverage),expected=protocol['total_timed_predictions'],
                 completed_methods=len(comparison),status='complete' if len(results)==protocol['total_jobs'] else 'incomplete')
    save(run/'progress.json',summary)
    (run/'summary_zh.txt').write_text('E4 '+protocol.get('measurement_scope','full_validation_no_sampling')+' 状态：'+summary['status']+'\n已完成任务：%s/%s；预测：%s/%s；完整方法行：%s/%s。\n'%(len(results),protocol['total_jobs'],summary['processed'],summary['expected'],len(comparison),len(protocol['methods']))+
        '请以 coverage.csv 查看逐任务状态；完整任务结果见 latency_by_dataset.csv；方法四数据集全部完成后才进入 comparison.csv。\n精度与计时来自同一轮预测。缓存 CorrCLIP 和中心裁剪 CLIPtrase 必须按各自 scope 单列，不作无条件端到端/全图对比。\n')
    return summary

def run_queue(run):
    lock=(run/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((run/'protocol.json').read_text());failures=[]
    for method in protocol['methods']:
        for ds in protocol['datasets']:
            out=run/'jobs'/method/ds;out.mkdir(parents=True,exist_ok=True)
            for attempt in range(1,3):
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
                if method=='freeda':env['E4_PYTHON']='/data/tianyi/conda_envs/freeda-official-py38-t112b/bin/python'
                else:env.pop('E4_PYTHON',None)
                command=['bash',str(run/'code/python_e4'),str(run/'code/measure_e4_full.py'),'--run',str(run),'--method',method,'--dataset',ds]
                log=out/('worker_'+time.strftime('%Y%m%dT%H%M%S')+'_a'+str(attempt)+'.log')
                with log.open('w') as f:
                    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
                    save(run/'queue_status.json',dict(status='running',pid=os.getpid(),worker_pid=child.pid,method=method,dataset=ds,attempt=attempt,log=str(log),updated_utc=utc()))
                    while child.poll() is None:
                        summarize(run);time.sleep(10)
                    code=child.returncode
                print(method,ds,'exit',code,flush=True)
                if code==0 and (out/'result.json').exists():break
                with (run/'failure_history.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),method=method,dataset=ds,attempt=attempt,exit=code,log=str(log)))+'\n')
            else:failures.append([method,ds])
            summarize(run)
    result=summarize(run)
    save(run/'queue_status.json',dict(status=result['status'],pid=os.getpid(),failed_jobs=failures,updated_utc=utc()))
    if failures:raise RuntimeError('Jobs still incomplete: '+str(failures))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','status','summarize']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    if args.action=='prepare':prepare(run)
    elif args.action=='run':run_queue(run)
    else:print(json.dumps(summarize(run),indent=2))
if __name__=='__main__':main()
