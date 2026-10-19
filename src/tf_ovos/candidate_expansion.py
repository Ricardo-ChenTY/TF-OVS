"""Controlled nested candidate expansion for the four SAM proposal adapters.

Base strings/IDs and GT stay fixed. SAM masks, crop features and compositing
are shared across every candidate level. Timing here is proposal-only and is
not a cross-method E4 comparison. --prepare-only needs no model dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from tf_ovos.adapters.registry import ADAPTERS
from tf_ovos.data import load_manifest, read_vocab, write_jsonl
from tf_ovos.metrics import evaluate_semantic, load_label_map

METHODS = ('sam_amg_clip', 'sam_amg_siglip', 'dinov2_sam_clip', 'dinov2_sam_siglip')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def nested_vocabularies(base: list[str], pool: list[str], sizes: list[int], seed: int):
    if not base or len(set(base)) != len(base):
        raise ValueError('Base vocabulary must be nonempty and contain no exact duplicates')
    seen = set(base)
    extras = []
    for name in pool:
        if name not in seen:
            seen.add(name)
            extras.append(name)
    random.Random(seed).shuffle(extras)
    sizes = sorted(set([len(base), *sizes]))
    if sizes[0] < len(base) or sizes[-1] > len(base)+len(extras):
        raise ValueError(f'Sizes must be between {len(base)} and {len(base)+len(extras)}')
    # Exact duplicate strings only: no synonym crosswalk, no base-name rewriting.
    all_names = base+extras
    return {size: all_names[:size] for size in sizes}


def vision_cache_fingerprint(config: dict) -> str:
    # Candidate text changes with seed/order; vision features must not depend on it.
    vision_config = {key: value for key, value in config.items() if key not in {'text_features_sha256', 'num_classes', 'task', 'prompt_template'}}
    return hashlib.sha256(json.dumps(vision_config, sort_keys=True).encode()).hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _prepare_regions(adapter, sample):
    with Image.open(sample.image_path) as image:
        image = image.convert('RGB')
    masks = sorted(adapter.mask_generator.generate(np.asarray(image)), key=lambda m:m['area'], reverse=True)
    crops = []
    for m in masks:
        x,y,w,h = map(int,m['bbox'])
        crop = image.crop((x,y,min(x+w,image.width),min(y+h,image.height)))
        crops.append(crop if min(crop.size)>0 else image)
    features = adapter._encode_crops(crops or [image]).float().cpu().numpy()
    owner = np.full((image.height,image.width),-1,dtype=np.int32)
    confidence = np.full(owner.shape,-1,dtype=np.float32)
    dino = adapter._dinov2_mask_scores(image,masks) if hasattr(adapter,'_dinov2_mask_scores') and masks else np.zeros(len(masks))
    for i,m in enumerate(masks):
        score = float(m['predicted_iou'])*float(m['stability_score'])*(1+getattr(adapter,'dinov2_weight',0)*float(dino[i]))
        update = m['segmentation'] & (score>confidence)
        owner[update] = i
        confidence[update] = score
    if not masks:
        owner[:] = 0  # Same full-image fallback as adapters.
    packed = np.stack([np.packbits(m['segmentation'].reshape(-1)) for m in masks]) if masks else np.empty((0,0),dtype=np.uint8)
    return dict(features=features, owner=owner, packed_masks=packed,
                shape=np.array(owner.shape), num_masks=np.array(len(masks)),
                dino_scores=dino, sam_confidences=np.array([float(m['predicted_iou'])*float(m['stability_score']) for m in masks]))


def run(args):
    base = read_vocab(args.base_vocab)
    levels = nested_vocabularies(base,read_vocab(args.distractor_vocab),args.sizes,args.seed)
    out = args.out.resolve()
    # A rerun gets a fresh directory; never mix corrected artifacts with old ones.
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f'Output must be empty: {out}')
    plan = dict(method=args.method, seed=args.seed, num_eval_classes=len(base),
                candidate_sizes=list(levels), base_vocab_sha256=sha256(args.base_vocab),
                distractor_vocab_sha256=sha256(args.distractor_vocab),
                gt_policy='unchanged_compact_gt', candidate_policy='unchanged_base_prefix_exact_string_dedup_only',
                matching='dense_class_region_max_iou_v2', status='prepared_not_run')
    for size,names in levels.items():
        d = out/f'candidates_{size}'
        d.mkdir()
        (d/'vocabulary.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
        _json(d/'mapping.json',[dict(candidate_id=i,name=name,eval_id=i if i<len(base) else None) for i,name in enumerate(names)])
    _json(out/'plan.json',plan)
    if args.prepare_only:
        print(f'Prepared {list(levels)} with {len(base)} unchanged base classes: {out}')
        return
    if args.manifest is None:
        raise ValueError('--manifest is required for inference')
    samples = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        samples = samples[:args.limit]
    if not samples or len({s.image_id for s in samples})!=len(samples):
        raise ValueError('Manifest must be nonempty with unique image IDs')
    for s in samples:
        if not s.image_path.is_file() or not s.mask_path.is_file():
            raise FileNotFoundError(f'Missing input: {s}')
    write_jsonl([dict(image_id=s.image_id,image_path=str(s.image_path.resolve()),mask_path=str(s.mask_path.resolve())) for s in samples],out/'manifest.jsonl')
    inputs = [dict(image_id=s.image_id,image_sha256=sha256(s.image_path),gt_sha256=sha256(s.mask_path)) for s in samples]
    _json(out/'input_hashes.json',inputs)
    import torch
    def sync():
        if torch.cuda.is_available(): torch.cuda.synchronize()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    start=time.perf_counter()
    adapter=ADAPTERS[args.method]()
    adapter.configure(task='semantic',num_classes=len(base))
    adapter._load()
    largest=max(levels)
    text=adapter._encode_text(levels[largest])
    sync()
    config=adapter.resolved_config()
    config['packages']={p:importlib.metadata.version(p) for p in ['torch','open_clip_torch','numpy','Pillow']}
    config['sam_checkpoint_sha256']=sha256(Path(adapter.sam_checkpoint))
    config['naming_checkpoint_sha256']={str(p):sha256(p) for p in Path(adapter.cache_dir).rglob('*') if p.is_file() and p.suffix in {'.safetensors','.bin','.pt','.pth'}}
    np.save(out/'text_features.npy',text.float().cpu().numpy())
    config['text_features_sha256']=sha256(out/'text_features.npy')
    code_paths=[Path(__file__),*sorted((Path(__file__).parent/'adapters').glob('*.py'))]
    config['source_sha256']={p.name:sha256(p) for p in code_paths}
    config['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    _json(out/'resolved_configuration.json',config)
    fingerprint=vision_cache_fingerprint(config)
    setup=time.perf_counter()-start
    cache_dir=(args.cache_dir or out/'region_cache').resolve(); cache_dir.mkdir(parents=True,exist_ok=True)
    for _ in range(args.warmup):
        _prepare_regions(adapter,samples[0])
    sync()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    gt_maps=[]; preds={size:[] for size in levels}; indices={size:[] for size in levels}
    timings=[]
    plan.update(status='running', processed_images=0, expected_images=len(samples))
    _json(out/'plan.json', plan)
    for index,(sample,input_info) in enumerate(zip(samples,inputs)):
        key=hashlib.sha256((fingerprint+input_info['image_sha256']).encode()).hexdigest()
        cache_path=cache_dir/f'{key}.npz'
        sync(); begin=time.perf_counter()
        hit=cache_path.exists()
        if hit:
            with np.load(cache_path,allow_pickle=False) as cache: regions={k:cache[k] for k in cache.files}
        else:
            regions=_prepare_regions(adapter,sample)
        sync(); vision_sec=time.perf_counter()-begin
        begin=time.perf_counter()
        if not hit: np.savez_compressed(cache_path,**regions)
        cache_export_sec=time.perf_counter()-begin
        features=torch.as_tensor(regions['features'],device=adapter.device,dtype=text.dtype)
        owner=regions['owner']; covered=owner>=0
        naming_sec=export_sec=0.
        for size in levels:
            sync(); begin=time.perf_counter()
            with torch.inference_mode(): best=(features@text[:size].T).argmax(dim=1).cpu().numpy()
            label=np.zeros(owner.shape,dtype=np.uint16 if size>256 else np.uint8)
            label[covered]=best[owner[covered]]
            sync(); naming_sec+=time.perf_counter()-begin
            begin=time.perf_counter()
            d=out/f'candidates_{size}'
            path=d/f'{index:06d}.png'; Image.fromarray(label).save(path)
            indices[size].append(dict(image_id=sample.image_id,mask_path=path.name,label=None,sha256=sha256(path)))
            preds[size].append(label)
            export_sec+=time.perf_counter()-begin
        gt_maps.append(load_label_map(str(sample.mask_path)))
        timings.append(dict(image_id=sample.image_id,cache_hit=hit,vision_sec=vision_sec,
                            cache_export_sec=cache_export_sec,naming_all_levels_sec=naming_sec,
                            prediction_export_all_levels_sec=export_sec,num_masks=int(regions['num_masks']),
                            crop_count=len(regions['features']),image_encode_batches=0 if hit else 1))
        plan.update(processed_images=index+1, last_image_id=sample.image_id)
        _json(out/'plan.json', plan)
        with (out/'timing_per_image.jsonl').open('a') as handle:
            handle.write(json.dumps(timings[-1])+'\n')
        print(f'{index+1}/{len(samples)} {sample.image_id}: masks={int(regions["num_masks"])} vision={vision_sec:.2f}s',flush=True)
    for size in levels:
        d=out/f'candidates_{size}'
        write_jsonl(indices[size],d/'predictions.jsonl')
        metrics=asdict(evaluate_semantic(gt_maps,preds[size],len(base),args.void_label,num_candidates=size))
        metrics.update(manifest_sha256=sha256(out/'manifest.jsonl'),vocab_sha256=sha256(d/'vocabulary.txt'),
                       sample_scope='pilot' if args.limit is not None else 'full_manifest')
        _json(d/'metrics.json',metrics)
    _json(out/'timing.json',dict(setup_sec=setup,warmup_images=args.warmup,
          scope='proposal_expansion_only_not_cross_method_E4',per_image=timings,
          peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None))
    plan.update(status='complete',processed_images=len(samples),sample_scope='pilot' if args.limit is not None else 'full_manifest')
    _json(out/'plan.json',plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,default='sam_amg_clip')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--base-vocab',required=True,type=Path)
    p.add_argument('--distractor-vocab',required=True,type=Path)
    p.add_argument('--sizes',nargs='+',required=True,type=int)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--limit',type=int)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--void-label',type=int,default=255)
    p.add_argument('--prepare-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__': main()
