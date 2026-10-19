"""One full E4 method/dataset per process, with atomic per-image resume."""
import argparse, gc, hashlib, io, json, os, platform, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from e4_full_common import (save,sha,utc,load_rows,open_db,validate_prefix,confusion_matrix,
                            pack_confusion,finish_job)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--method',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--job-dir',type=Path);p.add_argument('--stop-after',type=int)
    args=p.parse_args();run=args.run.resolve();protocol=json.loads((run/'protocol.json').read_text())
    if args.stop_after is not None and args.stop_after < 1:raise ValueError('--stop-after must be positive')
    if args.method not in protocol['methods'] or args.dataset not in protocol['datasets']:raise ValueError('Unknown job')
    source=json.loads((run/'source_sha256.json').read_text())
    for path,digest in source.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    spec=protocol['datasets'][args.dataset];manifest=run/spec['manifest']
    if sha(manifest)!=spec['sha256']:raise RuntimeError('Manifest changed')
    rows=load_rows(manifest);assert len(rows)==spec['images']
    warmup_rows=rows
    if spec.get('warmup_manifest'):
        warmup_path=run/spec['warmup_manifest']
        if sha(warmup_path)!=spec['warmup_sha256']:raise RuntimeError('Warmup manifest changed')
        warmup_rows=load_rows(warmup_path)
    output=(args.job_dir or run/'jobs'/args.method/args.dataset).resolve();output.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_sha256=sha(run/'protocol.json'),source_manifest_sha256=sha(run/'source_sha256.json'),
                  method=args.method,dataset=args.dataset,manifest_sha256=sha(manifest))
    db=open_db(output/'measurements.sqlite',identity);done=validate_prefix(db,rows,output)
    completed_file=output/'result.json'
    if completed_file.exists():
        if done!=len(rows):raise RuntimeError('Completed job has incomplete records')
        print('ALREADY COMPLETE',args.method,args.dataset,done);db.close();return
    def status(state,**extra):
        save(output/'status.json',dict(status=state,pid=os.getpid(),method=args.method,dataset=args.dataset,
                                     processed=done,expected=len(rows),updated_utc=utc(),**extra))
    status('loading');guard=None
    try:
        from e4_gpu_guard import GPUGuard
        guard=GPUGuard();guard.wait_idle(lambda ids:status('waiting_for_gpu',other_pids=ids))
        import torch
        import measure_e4 as engine
        engine.torch=torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(0);np.random.seed(0)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=protocol.get("cudnn_tf32",False)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        start=time.perf_counter()
        vocab=protocol.get('candidate_vocabulary');words=None
        if vocab:
            if sha(run/vocab['path'])!=vocab['sha256']:raise RuntimeError('Candidate vocabulary changed')
            words=(run/vocab['path']).read_text().splitlines()
            if len(words)!=vocab['classes']:raise RuntimeError('Candidate size differs')
        if args.method in engine.DENSE:
            vocab=protocol.get('candidate_vocabulary')
            if vocab and sha(run/vocab['path'])!=vocab['sha256']:raise RuntimeError('Candidate vocabulary changed')
            backend=engine.Dense(args.method,args.dataset,output,mask_rows=[],name_path=run/vocab['path'] if vocab else None,literal_candidates=protocol.get('literal_candidates',False))
            if vocab and int(backend.model.query_idx.max().item())+1!=vocab['classes']:raise RuntimeError('Candidate ID mismatch')
            if protocol.get('fail_on_pamr_fallback'):
                import logging
                class NoPAMRFallback(logging.Handler):
                    def emit(self,record):
                        if "Couldn't apply PAMR" in record.getMessage():raise RuntimeError('Native PAMR failed; refusing altered inference recipe')
                logging.getLogger().addHandler(NoPAMRFallback())
        elif args.method in ['cliptrase','freeda']:
            from e4_special_backends import Cliptrase,FreeDA
            backend=(Cliptrase if args.method=='cliptrase' else FreeDA)(args.method,args.dataset,output,words=words)
        else:backend=engine.Local(args.method,args.dataset,output,words=words)
        if protocol.get('pamr_channel_chunks') and getattr(backend.model,'pamr',None) is not None:
            from pamr_channel_chunks import ChannelChunkPAMR
            backend.model.pamr=ChannelChunkPAMR(backend.model.pamr)
        visual_cache=None
        if protocol.get('visual_stage_cache'):
            from candidate_stage_cache import attach
            visual_cache=attach(backend,args.method,args.dataset,protocol['visual_stage_cache']['path'],sha(run/'source_sha256.json'))
        torch.cuda.synchronize();setup=time.perf_counter()-start
        dtypes=Counter()
        for module in backend.modules:dtypes.update(str(t.dtype) for t in module.parameters())
        del module
        meta=json.loads(json.dumps(backend.meta,default=str))
        if args.method=='corrclip':
            meta.pop('mask_artifacts',None)
            meta['cache_policy']='native precomputed SAM2 masks loaded one image at a time before the online timer; cached-consumer supplemental row'
        scope='online_after_class_setup'
        if args.method=='corrclip':scope='cached_mask_consumer_NOT_end_to_end'
        if args.method=='corrclip_online':scope='online_SAM2_8_variant'
        geometry='native_336_center_crop' if args.method=='cliptrase' else 'original_frame'
        metadata=dict(identity,classes=spec['classes'],expected_images=len(rows),gpu=torch.cuda.get_device_name(0),
                      torch_version=torch.__version__,python_version=platform.python_version(),native_parameter_dtypes=dict(dtypes),
                      settings=meta,timing_scope=scope,quality_geometry=geometry,quality_source='same_timed_predictions',
                      protocol=protocol.get('measurement_scope','full_validation_no_sampling'),cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=False,setup_sessions='session_*.json',warmup_images_per_session=len(spec['warmup_ids']))
        if visual_cache is not None:
            metadata['visual_stage_cache']=visual_cache.report()
            metadata['timing_scope']='cached_execution_NOT_uncached_E4'
        save(output/'metadata.json',metadata)
        save(output/('session_'+str(os.getpid())+'.json'),dict(started_utc=utc(),resume_from=done,setup_seconds=setup,**metadata))
        gc.collect();torch.cuda.synchronize()
        pending={}
        if args.method=='corrclip':
            backend.model.generate_mask=lambda path:torch.from_numpy(pending['mask']).to(backend.model.device)
        def decode(row):
            blob=Path(row['image_path']).read_bytes()
            with Image.open(io.BytesIO(blob)) as im:rgb=im.convert('RGB')
            aux={};mask_blob=None
            if args.method=='corrclip':
                path=Path(meta['repo'])/meta['model']['instance_mask_path']/(Path(row['image_path']).stem+'.npz')
                if path.exists():
                    mask_blob=path.read_bytes()
                    with np.load(io.BytesIO(mask_blob),allow_pickle=False) as data:pending['mask']=data['instance_mask']
                else:pending['mask']=np.zeros((rgb.height,rgb.width),dtype=int)
                aux=dict(mask_path=str(path),missing_mask_fallback=mask_blob is None)
            return rgb,blob,mask_blob,aux
        by_id={row['image_id']:row for row in warmup_rows}
        status('warming_up')
        with torch.inference_mode():
            for image_id in spec['warmup_ids']:
                guard.wait_idle(lambda ids:status('waiting_for_gpu',other_pids=ids))
                row=by_id[image_id];rgb,_,_,_=decode(row);backend.infer(row,rgb);torch.cuda.synchronize()
        new_count=0
        for ordinal in range(done,len(rows)):
            row=rows[ordinal]
            while True:
                guard.wait_idle(lambda ids:status('waiting_for_gpu',other_pids=ids))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();guard.begin()
                read_start=time.perf_counter();rgb,blob,mask_blob,aux=decode(row)
                online_start=time.perf_counter()
                with torch.inference_mode():prediction=backend.infer(row,rgb)
                torch.cuda.synchronize();end=time.perf_counter()
                allocated=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
                other=guard.end()
                if not other:break
                with (output/'rejected_contention.jsonl').open('a') as f:
                    f.write(json.dumps(dict(image_id=row['image_id'],other_pids=other,utc=utc()))+'\n')
                status('waiting_for_gpu',other_pids=other)
            if hashlib.sha256(blob).hexdigest()!=row['image_sha256']:raise RuntimeError('Image content changed')
            gt_blob=Path(row['mask_path']).read_bytes()
            if hashlib.sha256(gt_blob).hexdigest()!=row['mask_sha256']:raise RuntimeError('GT content changed')
            with Image.open(io.BytesIO(gt_blob)) as im:gt=np.asarray(im).copy()
            if args.method=='cliptrase':gt=np.asarray(backend.native.gt_transform(336)(Image.fromarray(gt)))
            cm=confusion_matrix(gt,prediction,spec['classes'])
            if prediction.min()<0 or prediction.max()>65535:raise ValueError('Cannot export labels outside uint16')
            array=np.asarray(prediction,dtype=np.uint8 if prediction.max()<256 else np.uint16)
            target=output/'predictions'/('%06d.png'%ordinal);target.parent.mkdir(exist_ok=True)
            temp=target.with_suffix('.tmp');Image.fromarray(array).save(temp,format='PNG',compress_level=1);os.replace(temp,target)
            record=dict(image_id=row['image_id'],ordinal=ordinal,utc=utc(),session_pid=os.getpid(),
                        online_seconds=end-online_start,read_decode_seconds=online_start-read_start,
                        read_to_labels_seconds=end-read_start,peak_allocated_mib=allocated,peak_reserved_mib=reserved,
                        input_size=list(rgb.size),output_shape=list(prediction.shape),gpu=guard.telemetry(),
                        observed_other_gpu_pids=[],prediction_file=str(target.relative_to(output)),prediction_file_sha256=sha(target),
                        raw_prediction_sha256=hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
                        mask_sha256=hashlib.sha256(mask_blob).hexdigest() if mask_blob is not None else None,**aux)
            with db:
                db.execute('INSERT INTO images VALUES (?,?,?,?)',(row['image_id'],ordinal,json.dumps(record),pack_confusion(cm)))
            if visual_cache is not None and (ordinal%100==0 or ordinal+1==len(rows)):save(output/'visual_cache_usage.json',visual_cache.report())
            done=ordinal+1;new_count+=1;status('running',last_online_seconds=record['online_seconds'])
            if done%100==0:print(args.method,args.dataset,done,'/',len(rows),flush=True)
            if args.stop_after and new_count>=args.stop_after and done<len(rows):
                status('paused_for_verification');print('PAUSED',done,flush=True);return
        result=finish_job(db,rows,output,metadata);status('complete',online_mean_seconds=result['online_mean_seconds'])
        print('COMPLETE',args.method,args.dataset,done,result['online_mean_seconds'],flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:
        if guard is not None:guard.close()
        db.close()
if __name__=='__main__':main()
