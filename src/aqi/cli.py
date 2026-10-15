import argparse, json
from pathlib import Path
from .dataset import collect_samples, validate_mvtec, load_image
from .model import PatchCore
from .evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description="PatchCore-style MVTec AD inspection")
    p.add_argument("--data", default="data/mvtec_ad", help="MVTec AD root")
    p.add_argument("--category", default="metal_nut")
    p.add_argument("--output", default="artifacts/metal_nut")
    p.add_argument("--device", default=None)
    p.add_argument("--coreset-ratio", type=float, default=.1)
    p.add_argument("--limit-train", type=int, default=0); p.add_argument("--limit-test", type=int, default=0)
    args = p.parse_args()
    report = validate_mvtec(args.data, args.category); print(json.dumps(report, indent=2))
    if not report["valid"]: raise SystemExit("Dataset is unavailable or invalid. Run: python scripts/download_mvtec.py")
    train, test = collect_samples(args.data, args.category)
    if args.limit_train: train = train[:args.limit_train]
    if args.limit_test: test = test[:args.limit_test]
    model = PatchCore(args.device, coreset_ratio=args.coreset_ratio)
    model.fit([load_image(s.image, model.image_size) for s in train])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); model.save(out / "memory_bank.pt")
    result = evaluate(model, test, out); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
