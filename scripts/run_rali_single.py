"""Call the author's RALI demo model; --check-inputs uses only the standard library."""

import argparse
import contextlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def git_info(folder):
    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(folder), *arguments],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "status_before_run": git("status", "--short"),
                "diff_before_run": git("diff", "HEAD")}
    except (OSError, RuntimeError) as error:
        return {"commit": None, "error": str(error)}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e000_rali_single.json")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    upstream = resolve_path(config["upstream_repo"])
    weights = resolve_path(config["checkpoint_dir"])
    image_path = resolve_path(config["image_path"])
    required = [upstream / "demo_rali_score.py", weights / "ckpt.pt",
                weights / "pca.pkl", weights / "basis.npz",
                weights / "best" / "config.json",
                weights / "best" / "preprocessor_config.json", image_path]
    missing = [str(path) for path in required if not path.is_file()]
    model_files = list((weights / "best").glob("*.safetensors"))
    model_files += list((weights / "best").glob("pytorch_model*.bin"))
    if not model_files:
        missing.append(str(weights / "best") + " (model weights)")
    print(json.dumps({"python": sys.executable, "config": str(config_path),
                      "missing_inputs": missing}, ensure_ascii=False, indent=2))
    if missing:
        print("Inputs are not ready. No model was run and no record directory was created.")
        return 2
    if args.check_inputs:
        print("Path check passed. Dependencies, file contents and inference remain unverified.")
        return 0

    warmups = int(config.get("warmup_runs", 1))
    repeats = int(config.get("timed_runs", 3))
    if warmups < 0 or repeats < 1:
        raise ValueError("warmup_runs must be >= 0 and timed_runs must be >= 1")
    record_root = resolve_path(config["record_root"])
    try:
        record_root.relative_to(PROJECT / "experiments")
    except ValueError:
        raise ValueError("record_root must stay inside this project's experiments directory")

    own_git = git_info(PROJECT)
    upstream_git = git_info(upstream)
    run_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    destination = record_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"experiment_id": config["experiment_id"], "run_id": run_id,
                "status": "running", "started_at": datetime.now().astimezone().isoformat(),
                "python_executable": sys.executable, "python_version": sys.version,
                "project_git": own_git, "upstream_git": upstream_git,
                "image": str(image_path), "checkpoint_dir": str(weights)}
    save_json(destination / "config.json", config)
    save_json(destination / "metadata.json", metadata)
    try:
        # Imports are deliberately delayed: path checking needs no ML environment.
        import torch
        from PIL import Image
        sys.path.insert(0, str(upstream))
        from demo_rali_score import ClipScorer

        device_name = config.get("device", "cuda")
        if device_name not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA, but CUDA is unavailable. Check the interpreter.")
        device = torch.device(device_name)
        metadata["device"] = str(device)
        metadata["gpu"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
        metadata["torch_cuda_build"] = torch.version.cuda
        packages = ["torch", "torchvision", "transformers", "numpy", "Pillow",
                    "joblib", "scikit-learn", "scipy", "huggingface-hub"]
        metadata["packages"] = {}
        for package in packages:
            try:
                metadata["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                metadata["packages"][package] = None
        metadata["weight_files"] = {
            str(path.relative_to(weights)): path.stat().st_size
            for path in [weights / "ckpt.pt", weights / "pca.pkl", weights / "basis.npz", *model_files]
        }
        load_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(load_log), contextlib.redirect_stderr(load_log):
                scorer = ClipScorer(
                    ckpt_path=str(weights / "ckpt.pt"), clip_model=str(weights / "best"),
                    pca_path=str(weights / "pca.pkl"), basis_path=str(weights / "basis.npz"),
                    device=device,
                )
        finally:
            log_text = load_log.getvalue()
            (destination / "load.log").write_text(log_text, encoding="utf-8")
            print(log_text)
        if any(term in log_text.lower() for term in ("fallback", "failed to load", "using random")):
            raise RuntimeError("Author code reported a loading fallback. Inspect load.log before continuing.")
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        for _ in range(warmups):
            scorer.score_image(image)
        scores, seconds = [], []
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            score = float(scorer.score_image(image))
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            if not math.isfinite(score):
                raise ValueError("Non-finite model score")
            scores.append(score)
        save_json(destination / "prediction.json", {
            "image": str(image_path), "scores": scores,
            "score_mean": statistics.mean(scores), "score_spread": max(scores) - min(scores),
        })
        save_json(destination / "metrics.json", {
            "timed_runs": repeats, "seconds_per_run": seconds,
            "seconds_median": statistics.median(seconds),
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "timing_scope": "warm model, PIL image input, preprocessing and scoring; excludes disk read and model loading",
            "note": "Single-image smoke check; not a dataset benchmark. No MOS correlation measured.",
        })
        metadata["status"] = "completed"
        print(f"Score: {statistics.mean(scores):.6f}")
        print(f"Records: {destination}")
        return 0
    except Exception:
        metadata["status"] = "failed"
        error_text = traceback.format_exc()
        (destination / "error.log").write_text(error_text, encoding="utf-8")
        print(error_text, file=sys.stderr)
        print(f"Failure records: {destination}", file=sys.stderr)
        return 1
    finally:
        metadata["ended_at"] = datetime.now().astimezone().isoformat()
        save_json(destination / "metadata.json", metadata)


if __name__ == "__main__":
    raise SystemExit(main())
