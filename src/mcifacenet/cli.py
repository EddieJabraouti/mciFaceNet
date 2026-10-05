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
    extract = sub.add_parser("extract-video", help="Extract versioned AU/geometry signals and 42-feature summaries")
    extract.add_argument("--manifest", type=Path, default=Path("data/upstream/youtubepd/download_manifest.json"))
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--mediapipe-model", type=Path, default=Path("data/extractors/mediapipe/face_landmarker.task"))
    extract.add_argument("--window-seconds", type=float, default=5.0)
    extract.add_argument("--stride-seconds", type=float, default=5.0)
    extract.add_argument("--workers", type=int, default=2)
    extract.add_argument("--limit", type=int, help="Process the first N available clips for a pilot")
    fit = sub.add_parser("train", help="Train, evaluate, and export a new experiment")
    fit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fit.add_argument("--output", type=Path, required=True)
    youtube = sub.add_parser("youtube-train", help="Fit the original facial classifiers using only extracted YouTubePD windows")
    youtube.add_argument("--features-dir", type=Path, default=Path("data/processed/youtubepd_facial_v1"))
    youtube.add_argument("--manifest", type=Path, default=Path("data/upstream/youtubepd/download_manifest.json"))
    youtube.add_argument("--recipe", type=Path, default=Path("runs/facial_classifier_v1/protocol.json"))
    youtube.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict", help="Classify CSV rows with the 42 named features")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    fusion = sub.add_parser("fusion-train", help="Run the artificial label-paired facial/typing experiment")
    fusion.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fusion.add_argument("--typing-run", type=Path, required=True)
    fusion.add_argument("--face-model", type=Path, default=Path("runs/facial_classifier_v1/classifier.json"))
    fusion.add_argument("--output", type=Path, required=True)
    fixed = sub.add_parser("fusion-train-fixed", help="Train with one typist per facial participant and no window reassignment")
    fixed.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    fixed.add_argument("--typing-run", type=Path, required=True)
    fixed.add_argument("--face-model", type=Path, default=Path("runs/facial_classifier_v1/classifier.json"))
    fixed.add_argument("--previous-run", type=Path, default=Path("runs/facial_typing_fusion_v1"))
    fixed.add_argument("--output", type=Path, required=True)
    regularized = sub.add_parser("fusion-regularize", help="Compare dropout and sliding-window attention on existing fixed pairs")
    regularized.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    regularized.add_argument("--typing-run", type=Path, required=True)
    regularized.add_argument("--fixed-run", type=Path, default=Path("runs/facial_typing_fusion_fixed_v1"))
    regularized.add_argument("--output", type=Path, required=True)
    gallery = sub.add_parser("fusion-gallery", help="Train joint attention over ten typing embeddings and one facial token")
    gallery.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    gallery.add_argument("--typing-run", type=Path, required=True)
    gallery.add_argument("--previous-run", type=Path, default=Path("runs/facial_typing_fusion_regularized_v1"))
    gallery.add_argument("--output", type=Path, required=True)
    fused = sub.add_parser("fusion-predict", help="Classify aligned face/typing feature arrays without labels")
    fused.add_argument("--model", type=Path, required=True)
    fused.add_argument("--input", type=Path, required=True, help="NPZ with face [N,42] and typing [N,128], or [N,10,128] for gallery models")
    fused.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = {"verified_files": len(fetch_sources(args.data_dir))}
        elif args.command == "extract-video":
            from .video_extract import extract_videos
            result = extract_videos(args.manifest, args.output, args.mediapipe_model,
                                    args.window_seconds, args.stride_seconds, args.workers, args.limit)
        elif args.command == "train":
            from .train import train
            result = train(args.data_dir, args.output)
        elif args.command == "youtube-train":
            from .youtube_train import train_youtube
            result = train_youtube(args.features_dir, args.manifest, args.recipe, args.output)
        elif args.command == "fusion-train":
            from .fusion_train import train_fusion
            result = train_fusion(args.data_dir, args.typing_run, args.face_model, args.output)
        elif args.command == "fusion-train-fixed":
            from .fusion_fixed import train_fixed_fusion
            result = train_fixed_fusion(args.data_dir, args.typing_run, args.face_model, args.previous_run, args.output)
        elif args.command == "fusion-regularize":
            from .fusion_regularized_train import train_regularized_fusion
            result = train_regularized_fusion(args.data_dir, args.typing_run, args.fixed_run, args.output)
        elif args.command == "fusion-gallery":
            from .fusion_gallery_train import train_gallery_fusion
            result = train_gallery_fusion(args.data_dir, args.typing_run, args.previous_run, args.output)
        elif args.command == "fusion-predict":
            import numpy as np
            from .fusion import FusionClassifier
            from .fusion_data import sha256
            import torch
            bundle = torch.load(args.model, map_location="cpu", weights_only=True)
            if bundle.get("format_version") == 3:
                from .fusion_gallery import GalleryFusionClassifier
                model = GalleryFusionClassifier(bundle)
            elif bundle.get("format_version") == 2:
                from .fusion_regularized import RegularizedFusionClassifier
                model = RegularizedFusionClassifier(bundle)
            else:
                model = FusionClassifier(bundle)
            with np.load(args.input, allow_pickle=False) as features:
                extra = ({key: features[key] for key in ("sequence", "lengths") if key in features}
                         if bundle.get("format_version") == 2 else {})
                probabilities = model.predict_proba(features["face"], features["typing"], **extra)
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
        if args.command == "extract-video" and result["clips_failed"]:
            parser.exit(1, "Some clips failed extraction; see progress.json for details.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
