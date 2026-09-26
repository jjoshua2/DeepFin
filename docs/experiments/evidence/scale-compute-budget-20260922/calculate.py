"""Reproduce illustrative resource requirements; no jobs are launched."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = json.loads((HERE / "inputs.json").read_text())
assert abs(p["sf_fraction"] + p["bt4_fraction"] + p["ceres_fraction"] - 1) < 1e-12
n = p["target_positions"]
days = p["calendar_days"]
seconds = 86400
cpu = [
    {"availability": a, "required_accepted_sf_positions_per_second":
     n * p["sf_fraction"] / (days * a * seconds)}
    for a in p["cpu_availability"]
]
gpu = []
for reuse in p["label_reuse"]:
    bt4 = p["bt4_full_corpus_label_gpu_days"] * (1 - p["bt4_fraction"] if reuse else 1)
    ceres = p["ceres_full_corpus_label_gpu_days"] * (1 - p["ceres_fraction"] if reuse else 1)
    for availability in p["gpu_availability"]:
        remaining = days * availability - bt4 - ceres - p["one_epoch_training_gpu_days"]
        gpu.append({
            "reuse_generating_teacher_labels": reuse,
            "gpu_availability": availability,
            "remaining_label_gpu_days": bt4 + ceres,
            "training_gpu_days": p["one_epoch_training_gpu_days"],
            "available_generation_gpu_days": remaining,
            "minimum_aggregate_accepted_neural_positions_per_second":
                n * (p["bt4_fraction"] + p["ceres_fraction"]) / (remaining * seconds)
                if remaining > 0 else None,
            "resource_budget_feasible_before_generation": remaining > 0,
        })
print(json.dumps({"status": p["status"], "cpu": cpu, "gpu": gpu}, indent=2))
