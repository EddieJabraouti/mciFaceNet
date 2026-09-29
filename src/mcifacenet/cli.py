import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATA, fetch_sources, read_csv
from .model import FacialClassifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Facial-feature classification")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download and verify pinned public inputs")
    fetch.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fit = sub.add_parser("train", help="Train, evaluate, and export a new experiment")
    fit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fit.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict", help="Classify CSV rows with the 42 named features")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    fusion = sub.add_parser("fusion-train", help="Run the artificial label-paired facial/typing experiment")
    fusion.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fusion.add_argument("--typing-run", type=Path, required=True)
    fusion.add_argument("--face-model", type=Path, default=Path("runs/facial_classifier_v1/classifier.json"))
    fusion.add_argument("--output", type=Path, required=True)
    fused = sub.add_parser("fusion-predict", help="Classify aligned face/typing feature arrays without labels")
    fused.add_argument("--model", type=Path, required=True)
    fused.add_argument("--input", type=Path, required=True, help="NPZ with face [N,42] and typing [N,128]")
    fused.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = {"verified_files": len(fetch_sources(args.data_dir))}
        elif args.command == "train":
            from .train import train
            result = train(args.data_dir, args.output)
        elif args.command == "fusion-train":
            from .fusion_train import train_fusion
            result = train_fusion(args.data_dir, args.typing_run, args.face_model, args.output)
        elif args.command == "fusion-predict":
            import numpy as np
            from .fusion import FusionClassifier
            from .fusion_data import sha256
            model = FusionClassifier.load(args.model)
            with np.load(args.input, allow_pickle=False) as features:
                probabilities = model.predict_proba(features["face"], features["typing"])
            with args.output.open("x") as stream:
                json.dump({"classes": ["control", "impaired"], "model_sha256": sha256(args.model),
                           "probabilities": probabilities.tolist(),
                           "classifications": ["impaired" if p >= 0.5 else "control" for p in probabilities[:, 1]]},
                          stream, indent=2, allow_nan=False)
                stream.write("\n")
            result = {"predictions": len(probabilities), "output": str(args.output)}
        else:
            model = FacialClassifier.load(args.model)
            rows = read_csv(args.input)
            probabilities = model.predict_proba(rows)
            result = {"classes": model.artifact["classes"], "model_fingerprint": model.fingerprint,
                      "predictions": [{"row": index, "id": row.get("ID"),
                                       "p_control": p[0], "p_impaired": p[1],
                                       "classification": "impaired" if p[1] >= model.artifact["threshold"] else "control"}
                                      for index, (row, p) in enumerate(zip(rows, probabilities), 2)]}
            if args.output:
                with args.output.open("x") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                result = {"predictions": len(probabilities), "output": str(args.output)}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
