import pandas as pd

from src.end_to_end import build_end_to_end_evaluation


def test_end_to_end_evaluation_uses_common_cost_model() -> None:
    comparison = pd.DataFrame(
        [
            {
                "model": "feature_cost_v4",
                "benchmark": "vector_add",
                "target_candidate": True,
                "predicted_candidate": True,
                "gpu_runtime_ms": 1.0,
                "estimated_pim_time_ms": 0.25,
            },
            {
                "model": "feature_cost_v4",
                "benchmark": "gemm",
                "target_candidate": False,
                "predicted_candidate": False,
                "gpu_runtime_ms": 4.0,
                "estimated_pim_time_ms": 6.0,
            },
            {
                "model": "ai_only",
                "benchmark": "vector_add",
                "target_candidate": True,
                "predicted_candidate": True,
                "gpu_runtime_ms": 1.0,
                "estimated_pim_time_ms": "",
            },
            {
                "model": "ai_only",
                "benchmark": "gemm",
                "target_candidate": False,
                "predicted_candidate": True,
                "gpu_runtime_ms": 4.0,
                "estimated_pim_time_ms": "",
            },
        ]
    )

    evaluation = build_end_to_end_evaluation(comparison)
    gpu_only = evaluation[evaluation["model"] == "gpu_only"].iloc[0]
    feature_v4 = evaluation[evaluation["model"] == "feature_cost_v4"].iloc[0]
    ai_only = evaluation[evaluation["model"] == "ai_only"].iloc[0]

    assert gpu_only["total_runtime_ms"] == 5.0
    assert feature_v4["total_runtime_ms"] == 4.25
    assert feature_v4["false_offloads"] == 0
    assert ai_only["total_runtime_ms"] == 6.25
    assert ai_only["false_offloads"] == 1
