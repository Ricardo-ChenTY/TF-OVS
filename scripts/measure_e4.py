"""Isolated E4 online latency worker; native recipes, common decoded inputs, no vision cache."""
import argparse,contextlib,gc,hashlib,importlib,inspect,json,os,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/data/tianyi/TF-OVOS')
LEGACY=Path('/data/tianyi/TF-OVCOS/third_party/official_methods')
E4=ROOT/'runs/rebuttal/e4_20261009'
CACHE=Path('/data/sdd/tianyi/TF-OVOS-rebuttal/.cache/e4')
DENSE={'sclip':('SCLIP','clip_segmentor'),'naclip':('NACLIP','naclip'),'resclip':('ResCLIP','resclip_segmentor'),'scclip':('SC-CLIP','scclip_segmentor'),'proxyclip':('ProxyCLIP','proxyclip_segmentor'),'corrclip':('CorrCLIP','corrclip_segmentor'),'trident':('Trident','trident'),'cass':('CASS','cass_segmentor')}
DENSE['corrclip_online']=DENSE['corrclip']
DS={'voc20':('voc20','voc20.txt'),'context59':('context59','context_59.txt'),'ade150':('ade20k','ade20k_150.txt'),'coco_stuff171':('coco_stuff164k','coco_stuff_171.txt')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str)+'\n');tmp.replace(p)
def patch_clip_download(module):
 fn=module.load
 if 'download_root' in inspect.signature(fn).parameters:
  def load(*a,**kw):
   kw.setdefault('download_root',str(CACHE/'clip'));return fn(*a,**kw)
  module.load=load

class Local:
 def __init__(self,method,ds,out,words=None):
  from tf_ovos.adapters.registry import ADAPTERS
  self.a=ADAPTERS[method]();self.method=method;self.out=out;self.words=[s.strip() for s in (ROOT/'configs/vocab'/DS[ds][1]).read_text().splitlines() if s.strip() and not s.startswith('#')]
  if words is not None:self.words=list(words)
  self.a.configure(task='semantic',num_classes=len(self.words))
  if method.startswith('maskclip'):
   import clip;patch_clip_download(clip)
  self.a._load();self.text=self.a._encode_text(self.words)
  self.meta=self.a.resolved_config();self.meta['implementation']='repository_adapter';self.meta['class_vocabulary_sha256']=sha(ROOT/'configs/vocab'/DS[ds][1])
  if words is not None:self.meta['controlled_candidate_strings']=list(words)
  self.modules=[v for v in vars(self.a).values() if isinstance(v,torch.nn.Module)]
  if hasattr(self.a,'mask_generator'):self.modules.append(self.a.mask_generator.predictor.model)
 def infer(self,row,rgb):
  from tf_ovos.data import Sample
  a=self.a;sample=Sample(row['image_id'],Path(row['image_path']),Path(row['mask_path']))
  original_open=Image.open;original_save=Image.Image.save;capture={}
  def memory_open(path,*args,**kwargs):
   if str(path)==row['image_path']:return rgb.copy()
   return original_open(path,*args,**kwargs)
  def memory_save(im,*args,**kwargs):capture['prediction']=np.asarray(im).copy()
  try:
   Image.open=memory_open;Image.Image.save=memory_save
   if self.method.startswith('dinov2_'):a._predict_one_guided(sample,self.words,self.text,self.out,self.method)
   elif self.method=='maskclip':a._predict_with_text(sample,self.words,self.text,self.out)
   else:a._predict_one(sample,self.words,self.text,self.out)
  finally:Image.open=original_open;Image.Image.save=original_save
  return capture['prediction']

class Dense:
 def __init__(self,method,ds,out,mask_rows=None,name_path=None,literal_candidates=False):
  repo,module=DENSE[method];self.repo=LEGACY/repo;os.chdir(self.repo);sys.path.insert(0,str(self.repo))
  import clip;patch_clip_download(clip)
  if method in ['proxyclip','corrclip','corrclip_online','trident']:
   import open_clip
   original_create=open_clip.create_model
   def create(*a,**kw):
    kw.setdefault('cache_dir',str(CACHE/'clip'));return original_create(*a,**kw)
   open_clip.create_model=create
  native=importlib.import_module(module)
  os.environ['HF_ENDPOINT']='https://huggingface.co'
  if method=='corrclip_online':
   original_build=native.build_sam2
   def build(config,checkpoint,**kw):return original_build(config,str(CACHE/'weights/sam2_hiera_large.pt'),**kw)
   native.build_sam2=build
  import mmseg.datasets,mmseg.models
  from mmengine.config import Config
  from mmengine.registry import init_default_scope
  from mmengine.dataset import Compose,pseudo_collate
  from mmseg.registry import MODELS
  init_default_scope('mmseg')
  cfg=Config.fromfile(self.repo/'configs'/f'cfg_tfovos_{DS[ds][0]}.py')
  if name_path is not None:cfg.model['name_path']=str(Path(name_path).resolve())
  if method=='resclip':cfg.model.update(arch='wo_resi',attn_strategy='resclip',gaussian_std=5.)
  if method=='cass':cfg.model.update(pamr_steps=0,dataset=f'./configs/cfg_tfovos_{DS[ds][0]}.py')
  if method=='trident':cfg.model.update(sam_refinement=False,sam_ckpt=str(CACHE/'weights/sam_vit_b_01ec64.pth'))
  if method=='corrclip_online':cfg.model['mask_generator']='sam2'
  original_reader=None
  if literal_candidates:
   if name_path is None or not hasattr(native,'get_cls_idx'):raise RuntimeError('Literal candidate reader unavailable')
   original_reader=native.get_cls_idx
   def literal_reader(path):
    names=Path(path).read_text().splitlines();return names,list(range(len(names)))
   native.get_cls_idx=literal_reader
  try:self.model=MODELS.build(cfg.model).eval().cuda()
  finally:
   if original_reader is not None:native.get_cls_idx=original_reader
  self.modules=[self.model]
  self.method=method
  self.mask_info=[]
  if method=='corrclip':
   # Supplementary native cached mode; never presented as end-to-end latency.
   self.cached_masks={}
   if mask_rows is None:
    protocol=json.loads((E4/'protocol.json').read_text())['samples'][ds]
    mask_rows=protocol['warmup']+protocol['measure']
   for row in mask_rows:
    path=self.repo/cfg.model.instance_mask_path/(Path(row['image_path']).stem+'.npz')
    if path.exists():arr=np.load(path)['instance_mask']
    else:
     with Image.open(row['image_path']) as im:arr=np.zeros((im.height,im.width),dtype=int)
    self.cached_masks[row['image_path']]=arr
    self.mask_info.append(dict(image_id=row['image_id'],path=str(path),sha256=sha(path) if path.exists() else None,missing_fallback=not path.exists()))
   self.model.generate_mask=lambda path:torch.from_numpy(self.cached_masks[str(path)]).to(self.model.device)
  if method=='corrclip_online':self.modules.append(self.model.mask_generator.predictor.model)
  pipeline=cfg.test_dataloader.dataset.pipeline
  self.pipeline=Compose([dict(x) for x in pipeline if x['type'] not in ['LoadImageFromFile','LoadAnnotations']]);self.collate=pseudo_collate
  self.meta=dict(implementation='official_repository',repo=str(self.repo),model=dict(cfg.model),test_pipeline=[dict(x) for x in pipeline],config_sha256=sha(self.repo/'configs'/f'cfg_tfovos_{DS[ds][0]}.py'),class_vocabulary_sha256=sha(self.repo/cfg.model.name_path),output_geometry='native_preprocessing_and_postprocessing')
  if method=='corrclip':self.meta.update(timing_scope='cached_mask_consumer_only_NOT_end_to_end',vision_cache=True,mask_artifacts=self.mask_info,offline_mask_generation_seconds=None)
  if method=='corrclip_online':self.meta.update(timing_scope='end_to_end_native_online_SAM2_8_variant',vision_cache=False,quality_pairing='Not interchangeable with submitted pre-generated-mask E1 scores',sam2_checkpoint=str(CACHE/'weights/sam2_hiera_large.pt'),sam2_points_per_side=8)
 def infer(self,row,rgb):
  a=np.asarray(rgb)[:,:,::-1].copy();data=self.pipeline(dict(img=a,img_path=row['image_path'],ori_shape=a.shape[:2],img_shape=a.shape[:2]))
  original_open=Image.open
  def memory_open(path,*args,**kwargs):
   if str(path)==row['image_path']:return rgb.copy()
   return original_open(path,*args,**kwargs)
  try:
   Image.open=memory_open
   result=self.model.test_step(self.collate([data]))
  finally:Image.open=original_open
  return result[0].pred_sem_seg.data.squeeze(0).detach().cpu().numpy()

def main():
 global torch
 import torch
 worker_sha=sha(__file__)
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--limit',type=int,default=25);p.add_argument('--warmup',type=int,default=5);p.add_argument('--datasets',nargs='+',default=list(DS));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise RuntimeError('Refusing to overwrite an existing result')
 torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(0);np.random.seed(0)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('E4 requires CUDA')
 protocol=json.loads((E4/'protocol.json').read_text());results=[];settings=[]
 save(out/'status.json',dict(status='running',pid=os.getpid(),method=args.method))
 gpu=torch.cuda.get_device_properties(0)
 try:
  for ds in args.datasets:
   selected=protocol['samples'][ds];warm=selected['warmup'][:args.warmup];measure=selected['measure'][:args.limit]
   imgs={r['image_id']:Image.open(r['image_path']).convert('RGB') for r in warm+measure}
   t=time.perf_counter()
   if args.method in DENSE:backend=Dense(args.method,ds,out)
   elif args.method in ['cliptrase','freeda']:
    from e4_special_backends import Cliptrase,FreeDA
    backend=(Cliptrase if args.method=='cliptrase' else FreeDA)(args.method,ds,out)
   else:backend=Local(args.method,ds,out)
   torch.cuda.synchronize();setup=time.perf_counter()-t
   dtype_counts=Counter()
   for mod in backend.modules:
    dtype_counts.update({str(k):int(v) for k,v in Counter(str(p.dtype) for p in mod.parameters()).items()})
   del mod
   settings.append(json.loads(json.dumps(dict(dataset=ds,setup_seconds=setup,native_parameter_dtypes=dict(dtype_counts),**backend.meta),default=str)))
   gc.collect();torch.cuda.synchronize()
   save(out/'settings.json',settings)
   with torch.inference_mode():
    for row in warm:backend.infer(row,imgs[row['image_id']]);torch.cuda.synchronize()
    for index,row in enumerate(measure,1):
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
     pred=backend.infer(row,imgs[row['image_id']]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
     peak=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
     assert pred.ndim==2 and np.isfinite(pred).all()
     rec=dict(dataset=ds,image_id=row['image_id'],seconds=elapsed,peak_allocated_mib=peak,peak_reserved_mib=reserved,input_size=imgs[row['image_id']].size,output_shape=pred.shape,prediction_sha256=hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest())
     results.append(rec)
     with (out/'per_image.jsonl').open('a') as f:f.write(json.dumps(rec,default=str)+'\n')
     save(out/'status.json',dict(status='running',pid=os.getpid(),method=args.method,dataset=ds,processed=len(results),expected=args.limit*len(args.datasets),last_seconds=elapsed))
     if index%5==0:print(args.method,ds,index,'/',len(measure),'seconds',round(elapsed,4),flush=True)
   del backend;gc.collect();torch.cuda.empty_cache()
   with (out/'cleanup_memory.jsonl').open('a') as f:f.write(json.dumps(dict(dataset=ds,allocated_mib=torch.cuda.memory_allocated()/1024**2))+'\n')
  vals=np.array([r['seconds'] for r in results])
  result=dict(status='complete',method=args.method,images=len(results),mean_seconds=float(vals.mean()),median_seconds=float(np.median(vals)),p90_seconds=float(np.percentile(vals,90)),peak_allocated_mib=max(r['peak_allocated_mib'] for r in results),peak_reserved_mib=max(r['peak_reserved_mib'] for r in results),datasets={ds:dict(images=len([r for r in results if r['dataset']==ds]),mean_seconds=float(np.mean([r['seconds'] for r in results if r['dataset']==ds]))) for ds in args.datasets},setup_seconds=sum(x['setup_seconds'] for x in settings),torch_version=torch.__version__,gpu=gpu.name,gpu_total_memory=gpu.total_memory,protocol_sha256=sha(E4/'protocol.json'),worker_sha256=worker_sha,settings=settings)
  save(out/'result.json',result);save(out/'status.json',dict(status='complete',images=len(results),mean_seconds=result['mean_seconds']));print('COMPLETE',args.method,result['mean_seconds'],flush=True)
 except BaseException as e:
  save(out/'status.json',dict(status='failed',error=repr(e),processed=len(results)));raise
if __name__=='__main__':main()
