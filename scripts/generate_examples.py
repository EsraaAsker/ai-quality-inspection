#!/usr/bin/env python3
"""Generate compact, reproducible visual examples from a saved memory bank."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from aqi.dataset import collect_samples, load_image
from aqi.model import PatchCore


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', default='data/mvtec_ad'); p.add_argument('--category', default='metal_nut')
    p.add_argument('--artifact', default='artifacts/metal_nut/memory_bank.pt'); p.add_argument('--output', default='artifacts/metal_nut/examples')
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = PatchCore(); model.load_memory_bank(args.artifact)
    _, samples = collect_samples(args.data, args.category)
    metrics = json.loads((Path(args.artifact).parent / 'metrics.json').read_text()); threshold = metrics['image_threshold']
    scored = []
    for sample in samples:
        image = load_image(sample.image, model.image_size); pred = model.predict(image)
        if sample.label:
            mask = np.asarray(Image.open(sample.mask).convert('L').resize((model.image_size, model.image_size), Image.Resampling.NEAREST)) > 0
        else:
            mask = np.zeros((model.image_size, model.image_size), dtype=bool)
        scored.append((sample, image, pred, mask))
    normal = next(x for x in scored if not x[0].label)
    detected = max((x for x in scored if x[0].label and x[2].score >= threshold), key=lambda x: x[2].score)
    false_negative = min((x for x in scored if x[0].label and x[2].score < threshold), key=lambda x: x[2].score)
    chosen = [('normal', normal), ('detected_defect', detected), ('false_negative', false_negative)]
    manifest = []
    for name, (sample, image, pred, mask) in chosen:
        heat = np.asarray(Image.fromarray((pred.heatmap * 255).astype(np.uint8)).resize((model.image_size, model.image_size), Image.Resampling.BILINEAR)) / 255.0
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].imshow(image); ax[0].set_title(f'{name}\n{sample.image.name}')
        ax[1].imshow(heat, cmap='magma'); ax[1].set_title(f'heatmap\nscore={pred.score:.3f}')
        ax[2].imshow(image); ax[2].imshow(heat, cmap='jet', alpha=.45); ax[2].set_title('overlay')
        ax[3].imshow(mask, cmap='gray', vmin=0, vmax=1); ax[3].set_title('ground truth')
        for a in ax: a.axis('off')
        fig.tight_layout(); fig.savefig(out / f'{name}.png', dpi=140); plt.close(fig)
        manifest.append({'example': name, 'image': str(sample.image), 'defect_type': sample.defect_type,
                         'label': sample.label, 'score': pred.score, 'threshold': threshold,
                         'predicted_anomaly': bool(pred.score >= threshold)})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
