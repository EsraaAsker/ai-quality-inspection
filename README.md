# AI Quality Inspection

A GitHub-ready, CPU-compatible anomaly detection starter using **PyTorch**, a pretrained **ResNet18 PatchCore-style memory bank**, and the **MVTec AD** `metal_nut` category.

## Quickstart

```bash
cd ai-quality-inspection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_mvtec.py --output data/mvtec_ad
PYTHONPATH=src python -m aqi.cli --data data/mvtec_ad --category metal_nut --output artifacts/metal_nut
pytest -q
```

The CLI validates the dataset, fits on `train/good`, evaluates on the complete test split, writes `artifacts/metal_nut/metrics.json`, a memory bank, and up to 12 heatmap/overlay PNGs. **Metrics are produced by execution; no results are hard-coded.** Use `--limit-train` / `--limit-test` only for smoke tests.

## Demo and API

```bash
PYTHONPATH=src streamlit run src/aqi/streamlit_app.py
PYTHONPATH=src uvicorn aqi.api:app --reload
curl -F 'file=@path/to/image.png' http://127.0.0.1:8000/predict
```

Set `AQI_MEMORY_BANK` to use a different saved artifact. `/health` is available for liveness checks.

## Design notes

- **Model:** PatchCore-style nearest-neighbour detection. Frozen ImageNet ResNet18 layer2/layer3 patch embeddings are L2-normalized; a deterministic evenly spaced coreset keeps the memory bank CPU-friendly. The image score is the mean of the top 10% patch distances; the heatmap is the normalized patch distance map.
- **Metrics:** image-level AUROC and average precision use test good/defect labels. Pixel-level AUROC and average precision use MVTec ground-truth masks; good images receive all-zero masks. F1/confusion matrix are reported only at the transparent median-score threshold and are not a replacement for threshold-free metrics.
- **Data validation:** `validate_mvtec` checks the expected category folders and non-empty train/test image sets before model execution.
- **Reproducibility:** CPU is the default when CUDA is absent; GPU can be selected with `--device cuda`. Torchvision may download ResNet18 weights on first run.

## Structure

```text
src/aqi/       dataset validation, model, evaluation, API, Streamlit
scripts/       official MVTec archive downloader
tests/         focused unit/smoke tests
artifacts/     generated evaluation outputs (ignored by git)
```

## Dataset note

MVTec AD is hosted by MVTec and has its own terms. Review the dataset license before redistribution. The helper defaults to a public `metal_nut` mirror because the previously documented official archive URL returned HTTP 404 during this run; use `--source official --url <current-url>` if you have a current official archive URL. The evaluation in this workspace used the mirror and validated 220 train-good and 115 test images.
