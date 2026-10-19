"""Rescore saved CLIPtrase crops against official ADE847 GT, without inference."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode
ROOT = Path('/data/tianyi/TF-OVOS')
NATIVE = Path('/data/tianyi/TF-OVCOS/third_party/official_methods/CLIPtrase')
OUT = ROOT / 'runs/rebuttal/cliptrase_native_crop_20261015'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize(hist):
    tp=np.diag(hist);gt=hist.sum(1);pr=hist.sum(0);union=gt+pr-tp
    iou=np.divide(tp,union,out=np.zeros(len(tp)),where=union>0)
    return dict(native_gt_supported_miou_percent=float(np.mean((tp/(union+.01))[gt>0])*100),
                gt_supported_miou_no_epsilon_percent=float(iou[gt>0].mean()*100),
                union_supported_miou_percent=float(iou[union>0].mean()*100),
                ziou_gt_percent=float((tp[gt>0]==0).mean()*100),
                ziou_union_percent=float((tp[union>0]==0).mean()*100),
                gt_supported_classes=int((gt>0).sum()),union_supported_classes=int((union>0).sum()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=NATIVE/'clip_self_correlation.py'
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gt_transform')
    ns=dict(Compose=Compose,Resize=Resize,CenterCrop=CenterCrop,NEAREST=InterpolationMode.NEAREST)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    transform=ns['gt_transform'](336)
    cfg=NATIVE/'configs/dataset_cfg.py'
    info=next(ast.literal_eval(n.value) for n in ast.parse(cfg.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dataset_info' for t in n.targets))
    names=info['ADEfull']['labels'];vocab=(ROOT/'configs/vocab/ade20k_847.txt').read_text().splitlines()
    assert len(names)==len(vocab)==847
    differences=[dict(index=i,native=a,benchmark=b) for i,(a,b) in enumerate(zip(names,vocab)) if a!=b]
    assert all(a.replace("'",'')==b.replace("'",'') for a,b in zip(names,vocab))
    manifest=ROOT/'data/manifests/ade20k847_official_ids_audit.jsonl'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()];assert len(rows)==2000
    corrected=np.zeros((847,847),np.int64);historical=corrected.copy();records=[]
    oldroot=Path(info['ADEfull']['gt_path']);predroot=Path('/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cliptrase/ade847')
    for i,r in enumerate(rows):
        stem=Path(r['image_id']).name;pp=predroot/(stem+'.png');gp=Path(r['mask_path'])
        pred=np.asarray(Image.open(pp)).astype(np.int64)
        assert pred.shape==(336,336) and pred.min()>=0 and pred.max()<847
        gt=np.asarray(transform(Image.open(gp))).astype(np.int64)
        assert gt.shape==pred.shape and np.all((gt<847)|(gt==65535))
        valid=gt<847;corrected+=np.bincount((gt[valid]*847+pred[valid]).ravel(),minlength=847**2).reshape(847,847)
        op=oldroot/(r['image_id']+'.tif')
        if not op.exists():op=oldroot/(stem+'.tif')
        rec=dict(image_id=r['image_id'],prediction_sha256=sha(pp),official_gt_sha256=sha(gp),historical_gt_available=op.exists())
        if op.exists():
            old=np.asarray(transform(Image.open(op))).astype(np.int64)
            valid=(old>=0)&(old<847);historical+=np.bincount((old[valid]*847+pred[valid]).ravel(),minlength=847**2).reshape(847,847)
            rec.update(historical_gt_sha256=sha(op),changed_crop_gt_pixels=int((gt!=old).sum()))
        records.append(rec)
        if (i+1)%250==0:print('ADE847',i+1,flush=True)
    np.save(OUT/'official_confusion.npy',corrected);np.save(OUT/'historical_confusion.npy',historical)
    result=dict(images=2000,geometry='native Resize(336, nearest) + CenterCrop(336) on GT; predictions unchanged',
                prediction_encoding='zero-based semantic IDs; no subtraction',official=summarize(corrected),historical=summarize(historical) if all(r['historical_gt_available'] for r in records) else None,
                changed_crop_gt_pixels=sum(x.get('changed_crop_gt_pixels',0) for x in records) if all(r['historical_gt_available'] for r in records) else None,
                changed_crop_gt_images=sum(x.get('changed_crop_gt_pixels',0)>0 for x in records) if all(r['historical_gt_available'] for r in records) else None,
                vocabulary_differences=differences,
                source_sha256={str(p):sha(p) for p in [source,cfg,manifest,Path(__file__)]},records=records)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']},indent=2),flush=True)
    all_results={'ade847':result['official']}
    all_counts={'ade847':2000}
    source_records={'ade847':records}
    for ds,stem,classes,folder,offset in [('context59','context59_val',59,'context59',1),('context459','context459_val',459,'context459',0),('ade150','ade20k150_val',150,'ade20k',1)]:
        mp=ROOT/'data/manifests'/(stem+'.jsonl');items=[json.loads(x) for x in mp.read_text().splitlines()]
        hist=np.zeros((classes,classes),np.int64);details=[]
        for i,r in enumerate(items):
            pp=predroot.parent/folder/(Path(r['image_id']).name+'.png')
            gp=(mp.parent/r['mask_path']).resolve()
            pred=np.asarray(Image.open(pp)).astype(np.int64)-offset
            gt=np.asarray(transform(Image.open(gp))).astype(np.int64)
            assert pred.shape==gt.shape==(336,336)
            assert pred.min()>=0 and pred.max()<classes
            void=65535 if ds=='context459' else 255
            assert np.all((gt<classes)|(gt==void))
            valid=gt<classes
            hist+=np.bincount((gt[valid]*classes+pred[valid]).ravel(),minlength=classes**2).reshape(classes,classes)
            details.append(dict(image_id=r['image_id'],prediction_sha256=sha(pp),gt_sha256=sha(gp)))
        np.save(OUT/(ds+'_confusion.npy'),hist)
        all_results[ds]=summarize(hist);all_counts[ds]=len(items);source_records[ds]=details
        print(ds,len(items),all_results[ds],flush=True)
    tables=[dict(dataset=k,images=all_counts[k],**v) for k,v in all_results.items()]
    with (OUT/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    delta={}
    for key in ['native_gt_supported_miou_percent','union_supported_miou_percent']:
        c=all_results['context59'][key];cl=all_results['context459'][key]
        a=all_results['ade150'][key];al=all_results['ade847'][key]
        delta[key]=dict(context_drop_pp=c-cl,ade_drop_pp=a-al,mean_drop_pp=((c-cl)+(a-al))/2,
                        context_relative_drop_percent=100*(c-cl)/c,ade_relative_drop_percent=100*(a-al)/a,
                        mean_pair_relative_drop_percent=50*((c-cl)/c+(a-al)/a))
    (OUT/'pair_drops.json').write_text(json.dumps(delta,indent=2)+'\n')
    (OUT/'input_prediction_hashes.json').write_text(json.dumps(source_records,indent=2)+'\n')
    print(json.dumps(delta,indent=2),flush=True)

if __name__=='__main__':main()
