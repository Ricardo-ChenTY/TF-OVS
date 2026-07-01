#!/usr/bin/env python3
"""Build paper-style qualitative segmentation grids from TF-OVOS outputs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_path: Path
    label: str | None


@dataclass(frozen=True)
class Prediction:
    mask_path: Path
    label: str | None


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    cand = base / p
    if cand.exists():
        return cand
    return ROOT / p


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, Sample]:
    base = path.parent
    out: dict[str, Sample] = {}
    for row in read_jsonl(path):
        image_id = str(row['image_id'])
        out[image_id] = Sample(
            image_id=image_id,
            image_path=resolve_path(row['image_path'], base),
            mask_path=resolve_path(row['mask_path'], base),
            label=row.get('label'),
        )
    return out


def load_predictions(path: Path) -> dict[str, Prediction]:
    base = path.parent
    out: dict[str, Prediction] = {}
    for row in read_jsonl(path):
        image_id = str(row['image_id'])
        out[image_id] = Prediction(
            mask_path=resolve_path(row['mask_path'], base),
            label=row.get('label'),
        )
    return out


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
        '/Library/Fonts/Times New Roman.ttf',
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_rgb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert('RGB')
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def mask_bool(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path)
    if mask.mode not in ('L', 'I', 'I;16'):
        mask = mask.convert('L')
    target_size = (target_hw[1], target_hw[0])
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def overlay_mask(image_path: Path, mask_path: Path, size: int, color: tuple[int, int, int], alpha: float) -> Image.Image:
    original = Image.open(image_path).convert('RGB')
    mask = mask_bool(mask_path, (original.height, original.width))

    img = original.copy()
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.Resampling.NEAREST)

    color_img = Image.new('RGB', img.size, color)
    masked = Image.composite(color_img, img, resized_mask)
    blended = Image.blend(img, masked, alpha)

    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(blended, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def mask_iou(pred_path: Path, gt_path: Path) -> float:
    with Image.open(gt_path) as gt_img:
        shape = (gt_img.height, gt_img.width)
    gt = mask_bool(gt_path, shape)
    pred = mask_bool(pred_path, shape)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 0.0


def draw_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) // 2
    y = box[1] + (box[3] - box[1] - th) // 2
    draw.text((x, y), text, fill='black', font=font)


def draw_row_label(canvas: Image.Image, text: str, y: int, row_h: int, label_w: int, font: ImageFont.ImageFont) -> None:
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tmp = Image.new('RGBA', (bbox[2] - bbox[0] + 10, bbox[3] - bbox[1] + 10), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((5, 5), text, fill='black', font=font)
    tmp = tmp.rotate(90, expand=True)
    canvas.paste(tmp, ((label_w - tmp.width) // 2, y + (row_h - tmp.height) // 2), tmp)


def parse_method(spec: str) -> tuple[str, Path]:
    if '=' in spec:
        name, path = spec.split('=', 1)
        return name, Path(path)
    path = Path(spec)
    return path.parent.parent.name, path


def pick_samples(samples: dict[str, Sample], image_ids: list[str], count: int, seed: int) -> list[Sample]:
    if image_ids:
        missing = [image_id for image_id in image_ids if image_id not in samples]
        if missing:
            raise SystemExit(f'Missing image ids in manifest: {missing[:5]}')
        return [samples[image_id] for image_id in image_ids]
    values = list(samples.values())
    random.Random(seed).shuffle(values)
    return values[:count]


def caption(label: str | None, ok: bool | None, iou: float | None, show_iou: bool) -> str:
    text = label or 'mask'
    if ok is True:
        text += '[OK]'
    elif ok is False:
        text += '[X]'
    if show_iou and iou is not None:
        text += f' {iou:.2f}'
    return text


def build(args: argparse.Namespace) -> None:
    samples = load_manifest(args.manifest)
    chosen = pick_samples(samples, args.image_id, args.count, args.seed)
    method_rows = [(name, load_predictions(path)) for name, path in map(parse_method, args.method)]

    row_labels = ['Input', 'GT'] + [name for name, _ in method_rows]
    label_w = 76
    cap_h = 34
    gap = 8
    row_h = args.cell + cap_h
    width = label_w + len(chosen) * args.cell + max(0, len(chosen) - 1) * gap
    height = len(row_labels) * row_h
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    row_font = load_font(24)
    cap_font = load_font(20)

    for row_idx, row_label in enumerate(row_labels):
        y = row_idx * row_h
        draw_row_label(canvas, row_label, y, row_h, label_w, row_font)
        for col_idx, sample in enumerate(chosen):
            x = label_w + col_idx * (args.cell + gap)
            if row_label == 'Input':
                tile = fit_rgb(sample.image_path, args.cell)
                text = ''
            elif row_label == 'GT':
                tile = overlay_mask(sample.image_path, sample.mask_path, args.cell, (230, 48, 36), args.alpha)
                text = caption(sample.label, None, None, False)
            else:
                preds = dict(method_rows)[row_label]
                pred = preds.get(sample.image_id)
                if pred is None:
                    tile = fit_rgb(sample.image_path, args.cell)
                    text = 'missing[X]'
                else:
                    tile = overlay_mask(sample.image_path, pred.mask_path, args.cell, (34, 190, 54), args.alpha)
                    ok = None
                    if pred.label and sample.label:
                        ok = pred.label == sample.label
                    text = caption(pred.label or sample.label, ok, mask_iou(pred.mask_path, sample.mask_path), args.show_iou)
            canvas.paste(tile, (x, y))
            if text:
                draw_center(draw, (x, y + args.cell, x + args.cell, y + row_h), text, cap_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--method', action='append', default=[], help='Row=path/to/predictions.jsonl')
    parser.add_argument('--image-id', action='append', default=[])
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--cell', type=int, default=180)
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--show-iou', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.method:
        raise SystemExit('Pass at least one --method Row=path/to/predictions.jsonl')
    build(args)


if __name__ == '__main__':
    main()
