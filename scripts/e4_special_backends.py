"""Native CLIPtrase and FreeDA interfaces for the isolated E4 worker."""
import contextlib,hashlib,io,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
LEGACY=Path('/data/tianyi/TF-OVCOS/third_party/official_methods')
CACHE=Path('/data/sdd/tianyi/TF-OVOS-rebuttal/.cache/e4')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Cliptrase:
 def __init__(self,method,ds,out,words=None):
  repo=LEGACY/'CLIPtrase';os.chdir(repo);sys.path.insert(0,str(repo))
  import clip_utils
  import clip_self_correlation as native
  self.native=native;self.ds={'voc20':'VOC20','context59':'PC59','ade150':'ADE150','coco_stuff171':'COCO171_val'}[ds]
  self.model,_=clip_utils.load('ViT-B/16',image_size=224,download_root=str(CACHE/'clip'))
  self.model=self.model.cuda().eval();self.modules=[self.model]
  if words is not None:native.dataset_info[self.ds]['labels']=list(words)
  with torch.no_grad():self.text=native.get_text_features(self.model,native.dataset_info[self.ds]['labels'])
  self.dummy_gt=torch.zeros((336,336),device='cuda',dtype=torch.int16)
  self.meta=dict(implementation='official_self_clip_with_IO_metrics_replaced',repo=str(repo),image_size=336,eps=1.1,min_samples=7,output_geometry='native_336_center_crop',source_sha256=sha(repo/'clip_self_correlation.py'),class_names=native.dataset_info[self.ds]['labels'],excluded='disk IO, GT evaluation, cached class text; native inference body unchanged',note='loader image_size=224, inference image_size=336, matching original evaluation queue')
 def infer(self,row,rgb):
  n=self.native;captured={}
  def capture(dataset,name,pred):captured['pred']=pred.detach().cpu().numpy()
  patches=dict(get_image_name_list=lambda *a:[row['image_id']],get_image_and_gt=lambda *a:(n._transform1(336)(rgb),self.dummy_gt),get_text_features=lambda *a:self.text,_save_tfovos_prediction=capture,scores=lambda *a:{},tqdm=lambda x:x)
  old={k:getattr(n,k) for k in patches}
  try:
   for k,v in patches.items():setattr(n,k,v)
   with contextlib.redirect_stdout(io.StringIO()):n.self_clip(self.model,self.ds,image_size=336,eps=1.1,min=7)
  finally:
   for k,v in old.items():setattr(n,k,v)
  return captured['pred']
class FreeDA:
 def __init__(self,method,ds,out,words=None):
  repo=LEGACY/'freeda/src';os.chdir(repo);sys.path.insert(0,str(repo))
  import open_clip
  original_create=open_clip.create_model_and_transforms
  def create(*a,**kw):
   kw.setdefault('cache_dir',str(CACHE/'clip'));return original_create(*a,**kw)
  open_clip.create_model_and_transforms=create
  from omegaconf import OmegaConf
  import mmcv
  from mmcv.parallel import collate,scatter
  from mmseg.datasets import DATASETS,PIPELINES
  from mmseg.datasets.pipelines import Compose
  import segmentation.datasets
  from models import build_model
  from segmentation.evaluation.freeda_builder import build_freeda_seg_inference
  if 'FloatImage' not in PIPELINES.module_dict:
   @PIPELINES.register_module()
   class FloatImage:
    def __call__(self,results):results['img']=results['img'].astype(np.float32);return results
  config_name={'voc20':'voc20','context59':'context59','ade150':'ade20k150','coco_stuff171':'coco_stuff171'}[ds]
  config_path=Path('/data/tianyi/TF-OVCOS/runs/freeda')/config_name/'config.json'
  config=OmegaConf.load(config_path)
  dskey={'voc20':'voc20','context59':'context59','ade150':'ade20k','coco_stuff171':'coco_stuff'}[ds]
  segpath=repo/config.evaluate[dskey];cfg=mmcv.Config.fromfile(str(segpath))
  cls=DATASETS.get(cfg.dataset_type)
  class_names=list(cls.CLASSES) if words is None else list(words)
  dataset=SimpleNamespace(dataset=SimpleNamespace(CLASSES=class_names,PALETTE=cls.PALETTE))
  model_start=time.perf_counter();model=build_model(config.model).cuda().eval();torch.cuda.synchronize();model_seconds=time.perf_counter()-model_start
  proto_start=time.perf_counter()
  with torch.no_grad():self.model=build_freeda_seg_inference(model,dataset,config,str(segpath)).cuda().eval()
  torch.cuda.synchronize();proto_seconds=time.perf_counter()-proto_start
  self.modules=[self.model];self.collate=collate;self.scatter=scatter
  self.pipeline=Compose([dict(x) for x in cfg.test_pipeline if x['type']!='LoadImageFromFile'])
  self.meta=dict(implementation='official_FreeDASegInference',repo=str(repo),model=OmegaConf.to_container(config.model,resolve=True),evaluate=OmegaConf.to_container(config.evaluate,resolve=True),test_pipeline=[dict(x) for x in cfg.test_pipeline],test_cfg=dict(cfg.test_cfg),class_names=class_names,model_load_seconds=model_seconds,class_text_and_prototype_retrieval_seconds=proto_seconds,config_sha256=sha(config_path),seg_config_sha256=sha(segpath),source_sha256=sha(repo/'models/freeda/freeda.py'),offline=dict(prebuilt_index=str(repo/config.model.collection_index),prebuilt_prototypes=str(repo/config.model.collection_embeddings),construction_seconds=None,construction_note='existing artifacts reused; original offline generation time unavailable; not zero'),output_geometry='native_slide_and_original_size_postprocess')
 def infer(self,row,rgb):
  a=np.asarray(rgb)[:,:,::-1].copy();data=self.pipeline(dict(img=a,filename=row['image_path'],ori_filename=Path(row['image_path']).name,ori_shape=a.shape,img_shape=a.shape,pad_shape=a.shape,scale_factor=1.0,img_norm_cfg=dict(mean=np.zeros(3,dtype=np.float32),std=np.ones(3,dtype=np.float32),to_rgb=False),img_fields=['img']))
  batch=self.scatter(self.collate([data],samples_per_gpu=1),[0])[0]
  return np.asarray(self.model(return_loss=False,rescale=True,**batch)[0])
