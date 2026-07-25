import subprocess
from pathlib import Path
from datetime import datetime


# =========================
# Basic paths
# =========================

BASE_DIR = Path(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara")

WEIGHTS_HBB = BASE_DIR / "weights" / "yolo11l.pt"
WEIGHTS_OBB = BASE_DIR / "weights" / "yolo11l-obb.pt"

REPORT_DIR = BASE_DIR / "hara_report"

EXPERIMENTS = [
    {
        "folder": "HBB5min",
        "model": WEIGHTS_HBB,
        "prefix": "hbb",
        "length": "5min",
    },
    {
        "folder": "OBB5min",
        "model": WEIGHTS_OBB,
        "prefix": "obb",
        "length": "5min",
    },
    {
        "folder": "OBB10min",
        "model": WEIGHTS_OBB,
        "prefix": "obb",
        "length": "10min",
    },
]


# =========================
# YOLO training parameters
# =========================

COMMON_ARGS = {
    "imgsz": 1280,
    "epochs": 300,
    "patience": 9999,
    "batch": 10,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 3.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 180,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 10,
    "seed": 0,
    "deterministic": True,
}

# Optional: put all outputs under one folder
PROJECT_DIR = REPORT_DIR / "runs_all_folds"

# Logs
LOG_DIR = REPORT_DIR / "training_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# If True, stop all remaining jobs when one training fails
STOP_ON_ERROR = True


def yolo_value(v):
    """Convert Python values to YOLO CLI-friendly values."""
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def build_command(model_path, yaml_path, run_name):
    cmd = [
        "yolo",
        "train",
        f"model={model_path}",
        f"data={yaml_path}",
        f"name={run_name}",
        f"project={PROJECT_DIR}",
    ]

    for k, v in COMMON_ARGS.items():
        cmd.append(f"{k}={yolo_value(v)}")

    return cmd


def run_job(cmd, log_file):
    print("=" * 80)
    print("Starting job:")
    print(" ".join(str(x) for x in cmd))
    print(f"Log file: {log_file}")
    print("=" * 80)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Start time: {datetime.now()}\n")
        f.write("Command:\n")
        f.write(" ".join(str(x) for x in cmd) + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        process.wait()

        f.write(f"\nEnd time: {datetime.now()}\n")
        f.write(f"Return code: {process.returncode}\n")

    return process.returncode


def main():
    jobs = []

    for exp in EXPERIMENTS:
        folder = exp["folder"]
        model = exp["model"]
        prefix = exp["prefix"]
        length = exp["length"]

        folder_path = REPORT_DIR / folder

        for hold_id in range(1, 7):
            yaml_path = folder_path / f"hold{hold_id}.yaml"
            run_name = f"{prefix}_hold{hold_id}_{length}"

            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML not found: {yaml_path}")

            cmd = build_command(model, yaml_path, run_name)
            log_file = LOG_DIR / f"{run_name}.log"

            jobs.append((run_name, cmd, log_file))

    print(f"Total jobs to run: {len(jobs)}")
    for i, (run_name, _, _) in enumerate(jobs, start=1):
        print(f"{i:02d}. {run_name}")

    print("\nTraining will start now.\n")

    failed_jobs = []

    for i, (run_name, cmd, log_file) in enumerate(jobs, start=1):
        print(f"\n\n########## Job {i}/{len(jobs)}: {run_name} ##########\n")

        return_code = run_job(cmd, log_file)

        if return_code != 0:
            print(f"\nJob failed: {run_name}, return code = {return_code}")
            failed_jobs.append(run_name)

            if STOP_ON_ERROR:
                print("STOP_ON_ERROR=True, stopping remaining jobs.")
                break
        else:
            print(f"\nJob finished successfully: {run_name}")

    print("\nAll requested jobs processed.")

    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            print(f"- {job}")
    else:
        print("\nNo failed jobs.")


if __name__ == "__main__":
    main()